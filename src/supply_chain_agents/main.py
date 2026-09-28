import os
import time

import dotenv
import pandas as pd
from smolagents import OpenAIServerModel

from supply_chain_agents.agents import Orchestrator
from supply_chain_agents.config import (
    DATA_DIR,
    PROJECT_ROOT,
    TEST_DIR,
    catalog_items,
    db_engine,
    product_prices,
)
from supply_chain_agents.config_logging import configure_logging, get_logger
from supply_chain_agents.context import Context
from supply_chain_agents.utils import (
    embed,
    generate_financial_report,
    get_min_stock_levels,
    init_database,
)

logger = get_logger(__name__)


# Run the test scenarios
def run_test_scenarios():
    """Run all sample customer requests through the multi-agent workflow"""
    # Initialize shared context and Logger
    context = Context(
        db_engine=db_engine,
        catalog_items=catalog_items,
        product_prices=product_prices,
        catalog_embeddings=embed(catalog_items),
    )

    # Initializing Database
    init_database(db_engine=context.db_engine)
    context.min_stock_levels = get_min_stock_levels(db_engine=context.db_engine)
    try:
        quote_requests_sample = pd.read_csv(
            os.path.join(DATA_DIR, "quote_requests_sample.csv")
        )
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        logger.error("Error loading test data: %s", e)
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(db_engine, initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    # Load environment variables for the API key
    dotenv.load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Initialize the model with the API key
    model = OpenAIServerModel(
        model_id="gpt-4o-mini",
        api_key=openai_api_key,
        temperature=0.0,
        parallel_tool_calls=False,
    )

    # Create the orchestrator
    orchestrator = Orchestrator(model, context)

    results = []

    for idx, (_, row) in enumerate(quote_requests_sample.iterrows(), start=1):
        request_date = row["request_date"].strftime("%Y-%m-%d")

        logger.info("\n=== Request %s ===", idx)
        logger.info("Context: %s organizing %s", row["job"], row["event"])
        logger.info("Request Date: %s", request_date)
        logger.info("Cash Balance: $%.2f", current_cash)
        logger.info("Inventory Value: $%.2f", current_inventory)

        # Extract the request
        request_with_date = f"{row['request']} (Date of request: {request_date})"
        # Process the request with the orchestrator
        response = orchestrator.process_customer_order(request_with_date)

        # Update state with the orchestrator's response
        if context.financial_report:
            current_cash = context.financial_report.cash_balance
            current_inventory = context.financial_report.inventory_value

        logger.info("RESPONSE: %s", response)
        logger.info("Updated Cash: $%.2f", current_cash)
        logger.info("Updated Inventory: $%.2f", current_inventory)

        results.append(
            {
                "request_id": idx,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )
        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(db_engine, final_date)
    logger.info("===== FINAL FINANCIAL REPORT =====")
    logger.info("Final Cash: $%.2f", final_report["cash_balance"])
    logger.info("Final Inventory: $%.2f", final_report["inventory_value"])

    # Save results
    pd.DataFrame(results).to_csv(
        os.path.join(TEST_DIR, "test_results.csv"), index=False
    )
    return results


if __name__ == "__main__":
    configure_logging()
    run_test_scenarios()
