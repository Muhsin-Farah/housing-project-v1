"""Run the complete housing data pipeline."""

import sys
from pathlib import Path

src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from src.clean import main as clean
from src.adjust import main as adjust
from src.load import main as load_to_db
from src.plots import create_visualizations


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("LONDON HOUSING DATA PIPELINE")
    print("=" * 80)
    
    try:
        print("\n[1/4] Clean data...")
        clean()
        
        print("\n[2/4] Feature engineering & adjustments...")
        adjust()
        
        print("\n[3/4] Generate visualizations...")
        create_visualizations()
        
        print("\n[4/4] Load to SQLite...")
        load_to_db()
        
        print("\n✓ PIPELINE COMPLETE!\n")
    except Exception as e:
        print(f"\n✗ Failed: {e}\n")
        raise
