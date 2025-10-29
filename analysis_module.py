import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
from sklearn.neighbors import KernelDensity
from scipy import stats
import folium
from folium.plugins import MarkerCluster, HeatMap
import matplotlib.pyplot as plt
import json
import os
import tempfile
import datetime
from functools import lru_cache
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Konfigurasi OSMnx
try:
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.timeout = 300
except AttributeError:
    try:
        ox.config(use_cache=True, log_console=False, timeout=300)
    except:
        pass

def setup_osmnx_storage():
    """
    Setup storage locations yang aman untuk OSMnx
    """
    try:
        # Buat directory yang aman untuk cache dan data
        base_dir = os.path.expanduser("~/osmnx_data")  # User home directory
        cache_dir = os.path.join(base_dir, "cache")
        data_dir = os.path.join(base_dir, "data")
        
        # Create directories jika belum ada
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # Configure OSMnx dengan storage locations yang eksplisit
        try:
            ox.config(
                use_cache=True,
                cache_folder=cache_dir,
                data_folder=data_dir,
                log_console=True
            )
        except:
            pass
        
        print(f"✅ OSMnx storage configured:")
        print(f"   Cache: {cache_dir}")
        print(f"   Data: {data_dir}")
        return cache_dir, data_dir
        
    except Exception as e:
        print(f"❌ Error setting up storage: {e}")
        # Fallback ke temporary directory
        temp_dir = tempfile.gettempdir()
        cache_dir = os.path.join(temp_dir, "osmnx_cache")
        data_dir = os.path.join(temp_dir, "osmnx_data")
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        try:
            ox.config(
                use_cache=True,
                cache_folder=cache_dir,
                data_folder=data_dir,
                log_console=True
            )
        except:
            pass
        
        print(f"⚠️ Using temporary storage:")
        print(f"   Cache: {cache_dir}")
        print(f"   Data: {data_dir}")
        return cache_dir, data_dir

def validate_coordinates(df):
    """
    Validasi dan bersihkan koordinat yang bermasalah
    """
    print("🔍 Validating coordinates...")
    
    lat_col = 'Koordinat GPS - Lintang'
    lon_col = 'Koordinat GPS - Bujur'
    
    initial_count = len(df)
    
    # Remove invalid coordinates
    df = df.dropna(subset=[lat_col, lon_col])
    df = df[(df[lat_col] != 0) & (df[lon_col] != 0)]
    
    # Check for duplicate coordinates
    coords_before = len(df)
    df = df.drop_duplicates(subset=[lat_col, lon_col])
    coords_after = len(df)
    
    if coords_before != coords_after:
        print(f"⚠️ Removed {coords_before - coords_after} duplicate coordinates")
    
    # Check coordinate range (Indonesia bounds)
    indonesia_bounds = {
        'lat_min': -11.0, 'lat_max': 6.0,
        'lon_min': 95.0, 'lon_max': 141.0
    }
    
    valid_coords = (
        (df[lat_col] >= indonesia_bounds['lat_min']) & 
        (df[lat_col] <= indonesia_bounds['lat_max']) &
        (df[lon_col] >= indonesia_bounds['lon_min']) & 
        (df[lon_col] <= indonesia_bounds['lon_max'])
    )
    
    df = df[valid_coords]
    
    final_count = len(df)
    
    print(f"📊 Coordinate validation results:")
    print(f"   • Initial points: {initial_count}")
    print(f"   • Final valid points: {final_count}")
    print(f"   • Removed: {initial_count - final_count}")
    
    if final_count < 3:
        raise ValueError("Too few valid coordinates for analysis (minimum 3 required)")
    
    return df

def load_accident_data(excel_path):
    """
    Load dan bersihkan data kecelakaan dari file Excel
    """
    try:
        print("📊 Loading data kecelakaan...")
        df = pd.read_excel(excel_path)
        
        lat_col = 'Koordinat GPS - Lintang'
        lon_col = 'Koordinat GPS - Bujur'
        
        if lat_col not in df.columns or lon_col not in df.columns:
            raise ValueError(f"Kolom {lat_col} atau {lon_col} tidak ditemukan")
        
        # Parse tanggal jika ada
        if 'Tanggal Kejadian' in df.columns:
            df['Tanggal_Parsed'] = pd.to_datetime(df['Tanggal Kejadian'], errors='coerce')
            df['Tahun'] = df['Tanggal_Parsed'].dt.year
        
        # Bersihkan data
        df = validate_coordinates(df)
        
        print(f"✅ Data loaded: {len(df)} titik kecelakaan")
        return df
        
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def download_or_load_road_network_safe(area_name="Gowa, South Sulawesi, Indonesia", cache_dir=None, data_dir=None):
    """
    Download road network dengan storage yang aman
    """
    try:
        # Setup storage jika belum ada
        if cache_dir is None or data_dir is None:
            cache_dir, data_dir = setup_osmnx_storage()
        
        # Tentukan nama file cache yang aman
        safe_name = area_name.replace(",", "_").replace(" ", "_").replace("__", "_")
        cache_file = os.path.join(data_dir, f"{safe_name}_graph.graphml")
        
        print(f"🗂️ Graph cache file: {cache_file}")
        
        # Cek apakah file graf sudah ada dan bisa diakses
        if os.path.exists(cache_file) and os.access(cache_file, os.R_OK):
            print(f"🔄 Loading road network from cache: {cache_file}")
            try:
                G = ox.load_graphml(cache_file)
                print(f"✅ Successfully loaded from cache")
            except Exception as e:
                print(f"⚠️ Cache file corrupted, re-downloading: {e}")
                raise  # Will trigger download
        else:
            print(f"🌐 Downloading road network untuk {area_name}...")
            G = ox.graph_from_place(area_name, network_type='drive')
            
            # Coba save ke cache
            try:
                ox.save_graphml(G, cache_file)
                print(f"💾 Road network saved to cache: {cache_file}")
            except Exception as e:
                print(f"⚠️ Could not save to cache: {e}")
                print("Continuing without cache...")
        
        # Convert ke undirected
        G_undirected = G.to_undirected()
        
        print(f"✅ Road network ready: {len(G_undirected.nodes)} nodes, {len(G_undirected.edges)} edges")
        return G_undirected
        
    except Exception as e:
        raise Exception(f"Error loading road network with safe storage: {str(e)}")

def debug_distance_matrix(distance_matrix):
    """
    Debug distance matrix untuk memastikan ada variasi yang cukup (FIXED VERSION)
    """
    print(f"\n🔍 Distance Matrix Analysis:")
    print(f"   • Shape: {distance_matrix.shape}")
    
    # FIXED: Handle case where all distances are zero
    non_zero_distances = distance_matrix[distance_matrix > 0]
    
    if len(non_zero_distances) == 0:
        print("❌ CRITICAL ERROR: All distances are zero!")
        print("   • This indicates duplicate coordinates or calculation error")
        print("   • Falling back to basic statistics")
        print(f"   • Min distance: 0.00 meters")
        print(f"   • Max distance: {np.max(distance_matrix):.2f} meters")
        print(f"   • All values are: {np.unique(distance_matrix)}")
        return distance_matrix
    
    print(f"   • Min distance: {np.min(non_zero_distances):.2f} meters")
    print(f"   • Max distance: {np.max(distance_matrix):.2f} meters")
    print(f"   • Mean distance: {np.mean(non_zero_distances):.2f} meters")
    print(f"   • Std distance: {np.std(non_zero_distances):.2f} meters")
    
    # Check zero distances (should only be diagonal)
    zero_distances = np.sum(distance_matrix == 0)
    expected_zeros = distance_matrix.shape[0]  # Only diagonal should be zero
    
    if zero_distances > expected_zeros:
        print(f"⚠️ WARNING: {zero_distances - expected_zeros} unexpected zero distances found")
    
    # Check untuk duplicate points (same coordinates)
    unique_distances = len(np.unique(distance_matrix))
    total_distances = distance_matrix.shape[0] * distance_matrix.shape[1]
    print(f"   • Unique distance values: {unique_distances} of {total_distances} total")
    
    return distance_matrix

def euclidean_distance(lat1, lon1, lat2, lon2):
    """
    Calculate Euclidean distance between two points (fallback method)
    """
    try:
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371000  # Radius of earth in meters
        
        return r * c
    except:
        return 0

def calculate_distance_matrix_fixed(df, graph, max_workers=1):
    """
    FIXED: Perhitungan distance matrix yang benar dengan error handling
    """
    try:
        print("🔄 Calculating distance matrix with fixed implementation...")
        
        # Siapkan koordinat sebagai list of tuples
        coords = [(row['Koordinat GPS - Lintang'], row['Koordinat GPS - Bujur']) 
                  for idx, row in df.iterrows()]
        n_points = len(coords)
        
        print(f"📊 Processing {n_points} accident points...")
        
        if n_points == 0:
            raise ValueError("No valid coordinates found in data")
        
        # Pre-compute nearest nodes dengan error handling yang lebih baik
        print("📍 Finding nearest nodes for all points...")
        nearest_nodes = {}
        valid_points = 0
        
        for i, (lat, lon) in enumerate(coords):
            try:
                # Pastikan koordinat valid
                if np.isnan(lat) or np.isnan(lon):
                    nearest_nodes[i] = None
                    continue
                
                node = ox.distance.nearest_nodes(graph, X=lon, Y=lat)
                nearest_nodes[i] = node
                valid_points += 1
                
            except Exception as e:
                print(f"⚠️ Error finding nearest node for point {i} ({lat}, {lon}): {e}")
                nearest_nodes[i] = None
        
        print(f"✅ Found nearest nodes for {valid_points}/{n_points} points")
        
        if valid_points == 0:
            raise ValueError("No valid nearest nodes found - check if coordinates are within the area")
        
        # Initialize distance matrix
        distance_matrix = np.zeros((n_points, n_points))
        
        def calculate_batch_distances_fixed(batch_indices):
            """
            Fixed batch processing dengan robust error handling
            """
            batch_results = []
            
            for i in batch_indices:
                if nearest_nodes[i] is None:
                    # Fallback ke Euclidean untuk titik yang tidak valid
                    for j in range(n_points):
                        if i == j:
                            distance = 0
                        else:
                            lat1, lon1 = coords[i]
                            lat2, lon2 = coords[j]
                            distance = euclidean_distance(lat1, lon1, lat2, lon2)
                        batch_results.append((i, j, distance))
                    continue
                
                source_node = nearest_nodes[i]
                
                try:
                    # Compute shortest path lengths dari source ke semua target
                    lengths = nx.single_source_dijkstra_path_length(
                        graph, source_node, weight='length'
                    )
                    
                    for j in range(n_points):
                        if i == j:
                            distance = 0
                        elif nearest_nodes[j] is None:
                            # Fallback ke Euclidean
                            lat1, lon1 = coords[i]
                            lat2, lon2 = coords[j]
                            distance = euclidean_distance(lat1, lon1, lat2, lon2)
                        elif nearest_nodes[j] in lengths:
                            distance = lengths[nearest_nodes[j]]
                        else:
                            # Tidak ada path, gunakan Euclidean
                            lat1, lon1 = coords[i]
                            lat2, lon2 = coords[j]
                            distance = euclidean_distance(lat1, lon1, lat2, lon2)
                        
                        batch_results.append((i, j, distance))
                        
                except Exception as e:
                    print(f"⚠️ Error calculating distances from point {i}: {e}")
                    # Fallback ke Euclidean untuk semua pairs
                    for j in range(n_points):
                        if i == j:
                            distance = 0
                        else:
                            lat1, lon1 = coords[i]
                            lat2, lon2 = coords[j]
                            distance = euclidean_distance(lat1, lon1, lat2, lon2)
                        batch_results.append((i, j, distance))
            
            return batch_results
        
        # Bagi work ke batches
        batch_size = max(1, n_points // max_workers)
        batches = [list(range(i, min(i + batch_size, n_points))) 
                   for i in range(0, n_points, batch_size)]
        
        print(f"🔄 Processing {len(batches)} batches with {max_workers} workers...")
        
        total_pairs_processed = 0
        
        # Process batches
        for batch in tqdm(batches, desc="Processing batches"):
            try:
                batch_results = calculate_batch_distances_fixed(batch)
                
                # Fill distance matrix
                for i, j, distance in batch_results:
                    distance_matrix[i, j] = distance
                    total_pairs_processed += 1
                    
            except Exception as e:
                print(f"⚠️ Batch processing error: {e}")
                continue
        
        print(f"✅ Distance calculation complete: {total_pairs_processed} pairs processed")
        
        if total_pairs_processed == 0:
            raise ValueError("No distance pairs were successfully calculated")
        
        return distance_matrix
        
    except Exception as e:
        raise Exception(f"Error calculating distance matrix: {str(e)}")

def create_weight_matrix_with_validation(distance_matrix, coords):
    """
    Enhanced version dengan comprehensive validation (FIXED VERSION)
    """
    try:
        print("⚖️ Creating weight matrix with validation...")
        
        # Debug distance matrix terlebih dahulu
        distance_matrix = debug_distance_matrix(distance_matrix)
        
        n_points = len(coords)
        
        # 1. Validate distance matrix
        if distance_matrix.shape != (n_points, n_points):
            raise ValueError(f"Distance matrix shape {distance_matrix.shape} doesn't match coords {n_points}")
        
        # 2. Check for invalid distances
        if np.any(np.isnan(distance_matrix)) or np.any(np.isinf(distance_matrix)):
            print("⚠️ WARNING: Distance matrix contains NaN or Inf values")
            distance_matrix = np.nan_to_num(distance_matrix, nan=0.0, posinf=50000.0)
        
        # 3. Handle case where all distances are zero
        non_zero_distances = distance_matrix[distance_matrix > 0]
        if len(non_zero_distances) == 0:
            print("❌ CRITICAL: All distances are zero - using fallback uniform weights")
            sample_weights = np.ones(n_points)
            return sample_weights
        
        # 4. Ensure matrix is symmetric
        if not np.allclose(distance_matrix, distance_matrix.T, rtol=1e-10):
            print("⚠️ WARNING: Distance matrix is not symmetric, making it symmetric")
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
        
        # 5. Apply exponential decay dengan parameter adjustment
        max_distance = np.max(distance_matrix)
        print(f"📏 Max distance in matrix: {max_distance:.2f} meters")
        
        if max_distance > 0:
            # Adaptive max_distance berdasarkan data
            # Gunakan 95th percentile sebagai max_distance untuk menghindari outliers
            max_distance_adaptive = np.percentile(non_zero_distances, 95)
            print(f"📏 Using adaptive max distance (95th percentile): {max_distance_adaptive:.2f} meters")
            
            # FIXED: Handle case where adaptive max distance is 0
            if max_distance_adaptive > 0:
                normalized_distances = np.minimum(distance_matrix / max_distance_adaptive, 1.0)
            else:
                print("⚠️ WARNING: Adaptive max distance is 0, using original max distance")
                normalized_distances = np.minimum(distance_matrix / max_distance, 1.0)
        else:
            normalized_distances = distance_matrix
        
        # 6. Apply exponential decay dengan parameter yang lebih agresif
        decay_parameter = -5  # Lebih agresif dari -3 untuk memberikan variasi lebih besar
        weight_matrix = np.exp(decay_parameter * normalized_distances)
        
        # Set diagonal to 1 (self-weight)
        np.fill_diagonal(weight_matrix, 1.0)
        
        # 7. Convert matrix ke sample weights
        sample_weights = np.sum(weight_matrix, axis=1)
        
        # 8. Normalization with better approach
        if np.sum(sample_weights) > 0:
            # Preserve relative differences, don't force uniform mean
            sample_weights = sample_weights / np.sum(sample_weights) * n_points
        else:
            print("❌ ERROR: All sample weights are zero")
            sample_weights = np.ones(n_points)
        
        # 9. Apply minimum weight dengan nilai yang lebih kecil
        min_weight = 0.01  # Lebih kecil dari 0.1 untuk preservasi variasi
        sample_weights = np.maximum(sample_weights, min_weight)
        
        # 10. Final validation
        weight_range = np.max(sample_weights) - np.min(sample_weights)
        print(f"📊 Final weight statistics:")
        print(f"   • Range: {weight_range:.6f}")
        print(f"   • Ratio (max/min): {np.max(sample_weights)/np.min(sample_weights):.2f}")
        
        if weight_range < 0.001:
            print("⚠️ WARNING: Weight range is very small, may not provide significant KDE differences")
        
        print("✅ Weight matrix created with validation")
        return sample_weights
        
    except Exception as e:
        raise Exception(f"Error creating validated weight matrix: {str(e)}")

def perform_kde_analysis_with_optimal_params(df, sample_weights=None, bandwidth=0.008, threshold=1.96):
    """
    Perform KDE analysis dengan parameter optimal
    """
    try:
        print("🔥 Performing KDE analysis...")
        print(f"Parameters: bandwidth={bandwidth}, threshold={threshold}")
        
        coords = df[['Koordinat GPS - Lintang', 'Koordinat GPS - Bujur']].values
        
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        
        if sample_weights is not None:
            print("Using weighted KDE approach (network distances)")
            kde.fit(coords, sample_weight=sample_weights)
        else:
            print("Using spatial-only KDE approach")
            kde.fit(coords)
        
        coords_density = kde.score_samples(coords)
        density_exp = np.exp(coords_density)
        z_scores = stats.zscore(density_exp)
        
        hotspot_indices = np.where(z_scores > threshold)[0]
        
        hotspots = []
        for i, idx in enumerate(hotspot_indices):
            hotspots.append({
                'latitude': coords[idx][0],
                'longitude': coords[idx][1],
                'density': density_exp[idx],
                'z_score': z_scores[idx]
            })
        
        print(f"✅ KDE analysis complete: {len(hotspots)} hotspots identified")
        
        return hotspots, kde, coords_density, density_exp, z_scores
        
    except Exception as e:
        raise Exception(f"Error in KDE analysis: {str(e)}")

def filter_roads_for_hotspots(hotspot_coords, graph, radius_km=0.15):
    """
    Filter road segments untuk hotspot dengan radius yang ditentukan
    """
    try:
        print("🛣️ Filtering road segments near hotspots...")
        
        if len(hotspot_coords) == 0 or graph is None:
            return []
        
        # Simple distance calculation without scipy
        def simple_distance(p1, p2):
            return np.sqrt((p1[0] - p2)**2 + (p1[1] - p2[1])**2)
        
        road_segments = []
        radius_deg = radius_km / 111.0  # Convert km to degrees approximately
        
        for u, v, edge_data in graph.edges(data=True):
            lat_u, lon_u = graph.nodes[u]["y"], graph.nodes[u]["x"]
            lat_v, lon_v = graph.nodes[v]["y"], graph.nodes[v]["x"]
            
            midpoint = [(lat_u + lat_v) / 2, (lon_u + lon_v) / 2]
            
            # Check if road segment is near any hotspot
            for hotspot_coord in hotspot_coords:
                if simple_distance(midpoint, hotspot_coord) < radius_deg:
                    road_segments.append(((lat_u, lon_u), (lat_v, lon_v)))
                    break
        
        print(f"✅ Found {len(road_segments)} road segments near hotspots")
        return road_segments
        
    except Exception as e:
        print(f"⚠️ Warning: Could not find road segments: {str(e)}")
        return []

def map_color(value, vmin, vmax):
    """Map density value ke warna Viridis"""
    if vmax == vmin:
        normalized = 0.5
    else:
        normalized = (value - vmin) / (vmax - vmin)
    
    normalized = max(0, min(1, normalized))
    cmap = plt.cm.viridis
    rgba = cmap(normalized)
    
    return f'#{int(rgba[0] * 255):02x}{int(rgba[1] * 255):02x}{int(rgba[2] * 255):02x}'

def get_accident_color_and_radius(tingkat_kecelakaan):
    """Get color and radius berdasarkan tingkat kecelakaan"""
    if tingkat_kecelakaan:
        level = str(tingkat_kecelakaan).strip().lower()
        if level in ["ringan", "rendah", "kecil"]:
            return "green", 5, "Ringan"
        elif level in ["sedang", "menengah", "moderate"]:
            return "orange", 7, "Sedang"
        elif level in ["berat", "tinggi", "parah", "severe"]:
            return "red", 9, "Berat"
        else:
            return "green", 5, "Ringan"
    else:
        return "green", 5, "Ringan"

def create_geojson_output(df, hotspots, output_dir):
    """
    Create GeoJSON file untuk compatibility dengan app.py
    """
    try:
        print("📄 Creating GeoJSON output...")
        
        features = []
        
        # Add accident points
        for idx, row in df.iterrows():
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row['Koordinat GPS - Bujur'], row['Koordinat GPS - Lintang']]
                },
                "properties": {
                    "id": int(idx),
                    "type": "accident",
                    "tingkat": row.get('Tingkat Kecelakaan', 'Ringan'),
                    "tanggal": str(row.get('Tanggal Kejadian', '')) if pd.notna(row.get('Tanggal Kejadian', '')) else ''
                }
            }
            features.append(feature)
        
        # Add hotspots
        for idx, hotspot in enumerate(hotspots):
            feature = {
                "type": "Feature", 
                "geometry": {
                    "type": "Point",
                    "coordinates": [hotspot['longitude'], hotspot['latitude']]
                },
                "properties": {
                    "id": f"hotspot_{idx}",
                    "type": "hotspot",
                    "density": hotspot['density'],
                    "z_score": hotspot['z_score']
                }
            }
            features.append(feature)
        
        geojson_data = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "total_accidents": len(df),
                "total_hotspots": len(hotspots),
                "analysis_method": "KDE Analysis",
                "generated_at": datetime.datetime.now().isoformat()
            }
        }
        
        # Save GeoJSON
        geojson_filename = f"hotspot_analysis_{int(time.time())}.geojson"
        geojson_path = os.path.join(output_dir, geojson_filename)
        
        with open(geojson_path, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ GeoJSON saved: {geojson_path}")
        return geojson_path
        
    except Exception as e:
        print(f"⚠️ Warning: Could not create GeoJSON: {e}")
        return None

def create_visualization_with_accident_points_and_filters(df, hotspots, road_segments, kde_model, sample_weights=None, output_dir=tempfile.gettempdir()):
    """
    Visualisasi dengan hotspots, accident points, dan filter checkbox
    """
    try:
        print("🗺️ Creating visualization...")
        
        if len(hotspots) > 0:
            center_lat = np.mean([h['latitude'] for h in hotspots])
            center_lon = np.mean([h['longitude'] for h in hotspots])
        else:
            center_lat = df['Koordinat GPS - Lintang'].mean()
            center_lon = df['Koordinat GPS - Bujur'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        if len(hotspots) > 0:
            densities = [h['density'] for h in hotspots]
            vmin, vmax = min(densities), max(densities)
        else:
            vmin, vmax = 0, 1
        
        # LAYER 1: HOTSPOTS
        hotspot_group = folium.FeatureGroup(name="Hotspot Area")
        
        for idx, hotspot in enumerate(hotspots):
            color = map_color(hotspot['density'], vmin, vmax)
            
            popup_text = f"""
            🔥 Hotspot #{idx+1}
            Density: {hotspot['density']:.4f}
            Z-score: {hotspot['z_score']:.2f}
            Coordinates: ({hotspot['latitude']:.6f}, {hotspot['longitude']:.6f})
            Parameters: BW=0.008, Z=1.96
            Method: {'Network Distance' if sample_weights is not None else 'Spatial KDE'}
            Road radius: 150m
            """
            
            folium.CircleMarker(
                location=[hotspot['latitude'], hotspot['longitude']],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=popup_text,
                tooltip=f"Hotspot {idx+1}"
            ).add_to(hotspot_group)
        
        # LAYER 2: ACCIDENT POINTS
        accident_group = folium.FeatureGroup(name="Titik Kecelakaan")
        
        for idx, row in df.iterrows():
            tingkat_kecelakaan = row.get('Tingkat Kecelakaan', 'Ringan')
            color, radius, level_normalized = get_accident_color_and_radius(tingkat_kecelakaan)
            
            popup_text = f"""
            📍 Kecelakaan #{idx+1}
            Tingkat: {level_normalized}
            Koordinat: ({row['Koordinat GPS - Lintang']:.6f}, {row['Koordinat GPS - Bujur']:.6f})
            """
            
            if 'Tanggal Kejadian' in row and pd.notna(row['Tanggal Kejadian']):
                popup_text += f"Tanggal: {row['Tanggal Kejadian']}"
            
            folium.CircleMarker(
                location=[row['Koordinat GPS - Lintang'], row['Koordinat GPS - Bujur']],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=popup_text,
                tooltip=f"Kecelakaan {level_normalized}"
            ).add_to(accident_group)
        
        # LAYER 3: ROAD SEGMENTS
        road_group = folium.FeatureGroup(name="Jalan Rawan")
        
        coords = df[['Koordinat GPS - Lintang', 'Koordinat GPS - Bujur']].values
        
        for ((lat_u, lon_u), (lat_v, lon_v)) in road_segments:
            midpoint_coords = np.array([[(lat_u + lat_v) / 2, (lon_u + lon_v) / 2]])
            midpoint_density = kde_model.score_samples(midpoint_coords)
            midpoint_density_exp = np.exp(midpoint_density[0])
            color = map_color(midpoint_density_exp, vmin, vmax)
            
            folium.PolyLine(
                locations=[(lat_u, lon_u), (lat_v, lon_v)],
                color=color,
                weight=3,
                opacity=0.7,
                tooltip=f"Road density: {midpoint_density_exp:.4f}"
            ).add_to(road_group)
        
        # ADD LAYERS TO MAP
        hotspot_group.add_to(m)
        accident_group.add_to(m)
        road_group.add_to(m)
        
        # ADD LAYER CONTROL
        folium.LayerControl(position='topright', collapsed=False).add_to(m)
        
        # ADD CUSTOM FILTER CONTROLS
        filter_html = '''
        <div style="position: fixed; 
                    top: 10px; left: 60px; width: 300px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>🎛️ Analysis Controls</b></p>
        <p>📊 Hotspots: {}</p>
        <p>📍 Accident Points: {}</p>
        <p>🛣️ Road Segments: {}</p>
        <p>⚙️ Method: {}</p>
        </div>
        '''.format(
            len(hotspots),
            len(df),
            len(road_segments),
            'Network Distance KDE' if sample_weights is not None else 'Spatial KDE'
        )
        
        m.get_root().html.add_child(folium.Element(filter_html))
        
        # Save map
        map_filename = f"hotspot_analysis_{int(time.time())}.html"
        map_path = os.path.join(output_dir, map_filename)
        
        m.save(map_path)
        
        print(f"✅ Visualization saved: {map_path}")
        
        return map_path
        
    except Exception as e:
        raise Exception(f"Error creating visualization: {str(e)}")

def run_spatial_only_analysis(df, output_dir):
    """
    Fallback analysis using only spatial KDE (no network distances) - FIXED RETURN
    """
    print("🔄 Running spatial-only KDE analysis...")
    
    coords = df[['Koordinat GPS - Lintang', 'Koordinat GPS - Bujur']].values
    
    # Perform spatial KDE
    hotspots, kde_model, coords_density, density_exp, z_scores = perform_kde_analysis_with_optimal_params(
        df, sample_weights=None
    )
    
    # Create GeoJSON
    geojson_path = create_geojson_output(df, hotspots, output_dir)
    
    # Create visualization
    map_path = create_visualization_with_accident_points_and_filters(
        df, hotspots, [], kde_model, None, output_dir
    )
    
    print("✅ Spatial-only analysis completed")
    
    # Return both paths as expected by app.py
    return geojson_path, map_path

def run_analysis_safe(input_excel_path, output_dir=None, **kwargs):
    """
    FIXED: Main analysis function - RETURNS 2 VALUES as expected by app.py
    """
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    
    try:
        print("🚀 STARTING SAFE ANALYSIS WITH ENHANCED ERROR HANDLING")
        print("=" * 60)
        
        # Setup storage
        cache_dir, data_dir = setup_osmnx_storage()
        
        # 1. Load and validate data
        df = load_accident_data(input_excel_path)
        
        if len(df) < 5:
            print("⚠️ WARNING: Very few data points, using spatial-only analysis")
            return run_spatial_only_analysis(df, output_dir)
        
        # 2. Load road network
        try:
            G = download_or_load_road_network_safe(cache_dir=cache_dir, data_dir=data_dir)
        except Exception as e:
            print(f"⚠️ Road network loading failed: {e}")
            print("🔄 Falling back to spatial-only analysis")
            return run_spatial_only_analysis(df, output_dir)
        
        # 3. Calculate distance matrix with enhanced error handling
        try:
            distance_matrix = calculate_distance_matrix_fixed(df, G, max_workers=1)
            
            # Validate distance matrix
            if np.all(distance_matrix == 0):
                print("❌ Distance matrix contains only zeros - falling back to spatial analysis")
                return run_spatial_only_analysis(df, output_dir)
                
        except Exception as e:
            print(f"⚠️ Distance calculation failed: {e}")
            print("🔄 Falling back to spatial-only analysis")
            return run_spatial_only_analysis(df, output_dir)
        
        # 4. Continue with network analysis if distance matrix is valid
        coords = df[['Koordinat GPS - Lintang', 'Koordinat GPS - Bujur']].values
        
        try:
            sample_weights = create_weight_matrix_with_validation(distance_matrix, coords)
        except Exception as e:
            print(f"⚠️ Weight matrix creation failed: {e}")
            print("🔄 Falling back to spatial-only analysis")
            return run_spatial_only_analysis(df, output_dir)
        
        # 5. Perform KDE analysis
        hotspots, kde_model, coords_density, density_exp, z_scores = perform_kde_analysis_with_optimal_params(
            df, sample_weights
        )
        
        # 6. Filter road segments
        if len(hotspots) > 0:
            hotspot_coords = [(h['latitude'], h['longitude']) for h in hotspots]
            road_segments = filter_roads_for_hotspots(hotspot_coords, G)
        else:
            road_segments = []
        
        # 7. Create GeoJSON
        geojson_path = create_geojson_output(df, hotspots, output_dir)
        
        # 8. Create visualization
        map_path = create_visualization_with_accident_points_and_filters(
            df, hotspots, road_segments, kde_model, sample_weights, output_dir
        )
        
        print("✅ SAFE ANALYSIS COMPLETED SUCCESSFULLY")
        
        # Return both paths as expected by app.py
        return geojson_path, map_path
        
    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")
        print("🔄 Attempting fallback spatial analysis...")
        try:
            return run_spatial_only_analysis(df, output_dir)
        except:
            raise Exception(f"Both main and fallback analysis failed: {str(e)}")

# Legacy functions untuk backward compatibility
def run_analysis(input_excel_path, output_dir=None, **kwargs):
    """
    Legacy function - calls the safe version and returns 2 values
    """
    return run_analysis_safe(input_excel_path, output_dir, **kwargs)

def run_analysis_with_comprehensive_validation(input_excel_path, output_dir=None, **kwargs):
    """
    Legacy function wrapper - returns 2 values  
    """
    return run_analysis_safe(input_excel_path, output_dir, **kwargs)