# ShopSense: Customer Purchasing Behavior Analysis
# Initial setup: check for the retail dataset.

from pathlib import Path


def main():
    print("Welcome to ShopSense")

    # Look for the dataset in the same folder as this script.
    dataset_path = Path(__file__).resolve().parent / "Online Retail.xlsx"

    if dataset_path.is_file():
        print("Dataset found:", dataset_path.name)
    else:
        print("Please add 'Online Retail.xlsx' to the project folder.")

    print("Next step: load and explore the transaction data.")


if __name__ == "__main__":
    main()
