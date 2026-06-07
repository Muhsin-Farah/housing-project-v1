"""Generate visualizations: scatter plots and distributions."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def create_visualizations():
    """Create scatter plots and distribution plots for housing data."""
    
    df = pd.read_csv('./data/london_properties_adjusted.csv', parse_dates=['date'])
    print(f"Loaded {len(df):,} records")
    
    # Set style
    plt.rcParams['figure.figsize'] = (16, 12)
    
    fig = plt.figure()
    fig.suptitle('London Housing Data Analysis', fontsize=18, fontweight='bold', y=0.995)
    
    # 1. Price vs Age
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(df['property_age'], df['price_csi_adjusted'], alpha=0.3, s=10, color='steelblue')
    ax1.set_xlabel('Property Age (years)')
    ax1.set_ylabel('CSI-Adjusted Price (£)')
    ax1.set_title('Price vs Property Age')
    ax1.ticklabel_format(style='plain', axis='y')
    
    # 2. Price vs Distance to Centre
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(df['distance_to_centre'], df['price_csi_adjusted'], alpha=0.3, s=10, color='coral')
    ax2.set_xlabel('Distance to Charing Cross (km)')
    ax2.set_ylabel('CSI-Adjusted Price (£)')
    ax2.set_title('Price vs Distance to Central London')
    ax2.ticklabel_format(style='plain', axis='y')
    
    # 3. Price Distribution (original)
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(df['price'], bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Price (£)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Price Distribution (Skewed)')
    ax3.ticklabel_format(style='plain', axis='x')
    
    # 4. Log Price Distribution (normalized)
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(df['log_price_adjusted'], bins=100, color='lightgreen', edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Log Price (ln(£))')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Log Price Distribution (Normalized)')
    
    # 5. Property Type vs Price
    ax5 = plt.subplot(2, 3, 5)
    df_sample = df.sample(n=min(50000, len(df)))
    property_types = {
        'F': 'Flat',
        'T': 'Terraced',
        'S': 'Semi-Detached',
        'D': 'Detached',
        'O': 'Other'
    }
    df_sample['property_type_name'] = df_sample['property_type'].map(property_types)
    
    # Box plot manually
    for i, ptype in enumerate(['Flat', 'Terraced', 'Semi-Detached', 'Detached', 'Other']):
        data = df_sample[df_sample['property_type_name'] == ptype]['log_price_adjusted'].dropna()
        if len(data) > 0:
            bp = ax5.boxplot([data], positions=[i], widths=0.6)
    
    ax5.set_xticklabels(['Flat', 'Terraced', 'Semi-D', 'Detached', 'Other'])
    ax5.set_ylabel('Log Price (ln(£))')
    ax5.set_title('Price Distribution by Property Type')
    
    # 6. District Distribution (top 10)
    ax6 = plt.subplot(2, 3, 6)
    top_districts = df['district'].value_counts().head(10)
    ax6.barh(range(len(top_districts)), top_districts.values, color='mediumpurple')
    ax6.set_yticks(range(len(top_districts)))
    ax6.set_yticklabels(top_districts.index)
    ax6.set_xlabel('Number of Sales')
    ax6.set_title('Top 10 Districts by Transaction Count')
    
    plt.tight_layout()
    
    output = './data/housing_analysis.png'
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"Total records: {len(df):,}")
    print(f"\nPrice Statistics:")
    print(f"  Mean: £{df['price_csi_adjusted'].mean():,.0f}")
    print(f"  Median: £{df['price_csi_adjusted'].median():,.0f}")
    print(f"  Std Dev: £{df['price_csi_adjusted'].std():,.0f}")
    print(f"  Range: £{df['price_csi_adjusted'].min():,.0f} - £{df['price_csi_adjusted'].max():,.0f}")
    print(f"\nProperty Age:")
    print(f"  Mean: {df['property_age'].mean():.1f} years")
    print(f"  Median: {df['property_age'].median():.1f} years")
    print(f"  Range: {df['property_age'].min()}-{df['property_age'].max()} years")
    print(f"\nDistance to Centre:")
    print(f"  Mean: {df['distance_to_centre'].mean():.1f} km")
    print(f"  Median: {df['distance_to_centre'].median():.1f} km")
    print(f"\nProperty Types:")
    print(df['property_type'].value_counts())
    print("=" * 80)


if __name__ == "__main__":
    create_visualizations()
