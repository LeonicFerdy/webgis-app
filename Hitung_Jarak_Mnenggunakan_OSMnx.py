import os
import pandas as pd
import osmnx as ox
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from functools import lru_cache

# Cache untuk mempercepat pencarian node terdekat
@lru_cache(maxsize=None)
def nearest_node_cached(graph, lat, lon):
    return ox.distance.nearest_nodes(graph, X=lon, Y=lat)

# Fungsi untuk menghitung jarak antar pasangan titik
def calculate_distance_for_pair(graph, point1, point2):
    try:
        node1 = nearest_node_cached(graph, point1[0], point1[1])
        node2 = nearest_node_cached(graph, point2[0], point2[1])

        distance = nx.shortest_path_length(graph, source=node1, target=node2, weight='length', method='dijkstra')

        return {
            'Point 1': f"({point1[0]}, {point1[1]})",
            'Point 2': f"({point2[0]}, {point2[1]})",
            'Distance (meters)': distance
        }
    except nx.NetworkXNoPath:
        return {
            'Point 1': f"({point1[0]}, {point1[1]})",
            'Point 2': f"({point2[0]}, {point2[1]})",
            'Distance (meters)': None
        }
    except Exception as e:
        print(f"Error processing pair {point1} - {point2}: {e}")
        return None

# Fungsi untuk menghitung jarak dengan paralelisasi
def calculate_shortest_distances_parallel(data, graph, start_index=0, max_pairs=None):
    pairs = []
    total_pairs = len(data) * (len(data) - 1) // 2
    print(f"Expected Total Pairs: {total_pairs}")

    # Membatasi jumlah pasangan yang dihitung jika max_pairs diberikan
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            pair_index = (len(data) * i) + j - (i * (i + 1)) // 2
            if pair_index >= start_index and (max_pairs is None or pair_index < start_index + max_pairs):
                pairs.append((data.iloc[i], data.iloc[j]))

    print(f"Pairs to process starting from index {start_index}: {len(pairs)}")

    results = []
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                calculate_distance_for_pair,
                graph,
                (pair[0]['Koordinat GPS - Lintang'], pair[0]['Koordinat GPS - Bujur']),
                (pair[1]['Koordinat GPS - Lintang'], pair[1]['Koordinat GPS - Bujur'])
            ): pair for pair in pairs
        }

        for future in tqdm(as_completed(futures), total=len(pairs), desc="Processing pairs"):
            result = future.result()
            if result:
                results.append(result)

    print(f"Processed Total Pairs: {len(results)}")
    return pd.DataFrame(results)

# Memuat data dari file Excel
file_path = r"C:\Users\Asus\OneDrive\Documents\Skripsi\selected_columns.xlsx"
print("Loading data from Excel...")
data = pd.read_excel(file_path)

# Tidak menghapus data duplikat berdasarkan koordinat saja, tetapi tetap menghitung berdasarkan ID
print("Checking for duplicates based on coordinates and ID...")
print(f"Duplicate rows based on coordinates and ID: {data.duplicated(subset=['Koordinat GPS - Lintang', 'Koordinat GPS - Bujur', 'ID']).sum()}")

# Menyaring duplikasi berdasarkan Koordinat GPS dan ID
data = data.drop_duplicates(subset=['Koordinat GPS - Lintang', 'Koordinat GPS - Bujur', 'ID'], keep=False)
print(f"Total unique points after removing duplicate entries: {len(data)}")

# Tentukan path file untuk menyimpan dan memuat graf jalan
graph_file = "gowa_graph.graphml"

# Memeriksa apakah file graf sudah ada
if os.path.exists(graph_file):
    print("Loading road network from file...")
    graph = ox.load_graphml(graph_file)
else:
    print("Downloading road network for Gowa...")
    graph = ox.graph_from_place("Gowa, South Sulawesi, Sulawesi, Indonesia", network_type='drive')
    ox.save_graphml(graph, graph_file)
    print("Road network downloaded and saved to file.")

# Tentukan batas maksimal pasangan yang akan dihitung dalam satu proses
max_pairs_per_process = 600000

# Hitung untuk bagian pertama
print("Calculating shortest distances for part 1...")
distances_df_part1 = calculate_shortest_distances_parallel(data, graph, start_index=0, max_pairs=max_pairs_per_process)

# Simpan hasil bagian pertama
csv_output_file_part1 = r"C:\Users\Asus\OneDrive\Documents\Skripsi\shortest_distances_part1_ex.csv"
distances_df_part1.to_csv(csv_output_file_part1, index=False)
print(f"Data bagian pertama telah disimpan ke dalam {csv_output_file_part1}")

# Hitung untuk bagian kedua
print("Calculating shortest distances for part 2...")
distances_df_part2 = calculate_shortest_distances_parallel(data, graph, start_index=max_pairs_per_process, max_pairs=None)

# Simpan hasil bagian kedua
csv_output_file_part2 = r"C:\Users\Asus\OneDrive\Documents\Skripsi\shortest_distances_part2_ex.csv"
distances_df_part2.to_csv(csv_output_file_part2, index=False)
print(f"Data bagian kedua telah disimpan ke dalam {csv_output_file_part2}")
