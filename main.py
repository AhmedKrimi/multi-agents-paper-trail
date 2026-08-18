import os
import time
import dotenv
import pandas as pd
from config import db_engine, available_items, products_price, working_dir
from context import Context
from agents import Orchestrator
from smolagents import OpenAIServerModel
from utils import generate_financial_report, init_database

# Run the test scenarios


def run_test_scenarios():
    context = Context(db_engine, available_items,
                      products_price, items_sold=set())
    print("Initializing Database...")
    init_database(db_engine=context.db_engine)
    try:
        quote_requests_sample = pd.read_csv(
            os.path.join(working_dir, "quote_requests_sample.csv")
        )
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values(
            "request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime(
        "%Y-%m-%d")
    report = generate_financial_report(db_engine, initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    # Load environment variables for the API key
    dotenv.load_dotenv(dotenv_path=os.path.join(working_dir, ".env"))
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

        print(f"\n=== Request {idx} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Extract the request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        # Process the request with the orchestrator
        response = orchestrator.process_customer_order(request_with_date)

        # Update state with the orchestrator's response
        report = generate_financial_report(db_engine, request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"RESPONSE: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

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
    final_date = quote_requests_sample["request_date"].max().strftime(
        "%Y-%m-%d")
    final_report = generate_financial_report(db_engine, final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv(
        os.path.join(working_dir, "test_results.csv"), index=False
    )
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
