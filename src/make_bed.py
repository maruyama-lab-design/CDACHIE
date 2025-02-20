import argparse
import pandas as pd
import seaborn as sns

def csv_to_bed(csv_file, output_bed_file, name_column="CDACHIE", window_size=100000, num_clusters=6):
    """
    Function to convert a CSV file to a BED file.

    Args:
        csv_file (str): Input CSV file path.
        output_bed_file (str): Output BED file path.
        name_column (str): Name of the CSV column to use for the name column in the BED file.
                           Default is "CDACHIE".
        window_size (int): Window size. Default is 100000.
        num_clusters (int): Number of clusters. Default is 6.
    """
    df = pd.read_csv(csv_file)

    # Prepare color palette (using seaborn)
    colors = sns.color_palette("tab10")  # tab10 color palette
    colors_rgb = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]

    df["itemRgb"] = df[name_column].apply(lambda x: ",".join(map(str, colors_rgb[int(x) % len(colors_rgb)])))

    # Prepare columns needed for the BED file
    df["chromStart"] = df["pos"]
    df["chromEnd"] = df["pos"] + window_size

    df["score"] = "."
    df["strand"] = "."
    df["thickStart"] = df["chromStart"]
    df["thickEnd"] = df["chromEnd"]

    # Change name column to C1~C6
    df["name"] = df[name_column].apply(lambda x: f"C{int(x) + 1}")

    # Change chr column to "chr[number]" format
    df["chr"] = df["chr"].astype(str).apply(lambda x: f"chr{x}")

    # Output as BED file (BED9 format)
    df_bed = df[["chr", "chromStart", "chromEnd", "name", "score", "strand", "thickStart", "thickEnd", "itemRgb"]]
    df_bed.to_csv(output_bed_file, sep="\t", index=False, header=False)

    print(f"BED file saved to {output_bed_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV file to BED file.")
    parser.add_argument("output", help="Output BED file path")
    parser.add_argument("--csv", default="data/output/clusters.csv", help="Input CSV file path (default: data/output/clusters.csv)")
    parser.add_argument("--num_clusters", type=int, default=6, help="Number of clusters (default: 6)")
    args = parser.parse_args()

    csv_to_bed(args.csv, args.output, num_clusters=args.num_clusters)