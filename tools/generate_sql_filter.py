import csv
from pathlib import Path

class SQL_Where:

    @staticmethod
    def generate(outlier_path: Path) -> None:

        buildings = []
        with open(outlier_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'Building' in row:
                    buildings.append(row['Building'])
        
        if not buildings:
            print("No buildings found in the CSV.")
            return

        # Escape single quotes in building names for SQL
        escaped_buildings = [b.replace("'", "''") for b in buildings]
        
        # Create the IN clause
        in_clause = ", ".join([f"'{b}'" for b in escaped_buildings])
        where_clause = f"WHERE Building IN ({in_clause})"
        
        print("Generated SQL WHERE clause:")
        print("-" * 80)
        print(where_clause)
        print("-" * 80)
