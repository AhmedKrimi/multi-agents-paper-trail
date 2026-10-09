import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine

from supply_chain_agents.inventory import PAPER_CATALOG

# Absolute path of the project root directory
PROJECT_ROOT = Path(__file__).parents[2]
# Absolute path of data directory
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# Path to database
DB_DIR = os.path.join(DATA_DIR, "munder_difflin.db")
# Absolute path of test directory
TEST_DIR = os.path.join(PROJECT_ROOT, "tests")
# Path to test database
TEST_DB_DIR = os.path.join(TEST_DIR, "test_munder_difflin.db")
# Create a SQLite database
db_engine = create_engine(f"sqlite:///{DB_DIR}")
# Create a test SQLite database
test_db_engine = create_engine(f"sqlite:///{TEST_DB_DIR}")
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
