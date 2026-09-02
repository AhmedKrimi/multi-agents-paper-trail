from pathlib import Path

from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine

from inventory import PAPER_CATALOG

# Absolute path of the working directory
working_dir = Path(__file__).parent
# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")
# Extract catalog item names
catalog_items = [paper["inventory_name"] for paper in PAPER_CATALOG]
# Extract the price of each product in the catalog
product_prices = {
    paper["inventory_name"]: paper["unit_price"] for paper in PAPER_CATALOG
}
# Create embedder for name matching
embedder = SentenceTransformer("BAAI/bge-m3")

# Workflow configuration
MATCHING_THRESHOLD = 0.7
SUPPLIER_PRICE_FACTOR = 0.75
