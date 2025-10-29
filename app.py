# app.py - Versi Diperbaiki dengan Better Error Handling dan Compatibility

import os
import sys
import traceback
import tempfile
import shutil
import zipfile
import io
import datetime
from flask import (
    Flask, render_template, request,
    send_file, jsonify, url_for
)

# ========================================
# COMPATIBILITY LAYER UNTUK DEPENDENCIES
# ========================================

def setup_compatibility():
    """Setup compatibility untuk berbagai versi dependencies"""
    try:
        # Shapely compatibility
        try:
            from shapely.errors import TopologicalError
            print(" Using Shapely 2.0+ API")
        except ImportError:
            from shapely.geos import TopologicalError
            print(" Using Shapely 1.x API")
        
        # OSMnx compatibility
        try:
            import osmnx as ox
            try:
                # OSMnx 2.0+
                ox.settings.use_cache = True
                ox.settings.log_console = False
                print(f" OSMnx {ox.__version__} configured (2.0+ API)")
            except AttributeError:
                try:
                    # OSMnx 1.x
                    ox.config(use_cache=True, log_console=False)
                    print(f" OSMnx {ox.__version__} configured (1.x API)")
                except:
                    print("âš ï¸ OSMnx configuration failed, using defaults")
        except ImportError:
            print("âš ï¸ OSMnx not available - will use fallback methods")
        
        return True
    except Exception as e:
        print(f"âš ï¸ Compatibility setup warning: {e}")
        return True  # Continue anyway

# Setup compatibility
setup_compatibility()

# ========================================
# IMPORT ANALYSIS MODULE DENGAN FALLBACK
# ========================================

analysis_module_available = False
analysis_function = None

# Try multiple import strategies
import_strategies = [
    ("analysis_module", "run_analysis_safe"), 
    ("analysis_module", "run_analysis"),
    ("analysis_module", "run_analysis_with_comprehensive_validation"),
]

for module_name, function_name in import_strategies:
    try:
        module = __import__(module_name)
        analysis_function = getattr(module, function_name)
        analysis_module_available = True
        print(f" Successfully imported {function_name} from {module_name}")
        break
    except (ImportError, AttributeError) as e:
        print(f"âš ï¸ Failed to import {function_name} from {module_name}: {e}")

if not analysis_module_available:
    print("âŒ No analysis module available - creating simple fallback")
    
    def simple_fallback_analysis(input_excel_path, output_dir=None, **kwargs):
        """Simple fallback analysis jika module utama tidak tersedia"""
        import pandas as pd
        import json
        
        if output_dir is None:
            output_dir = tempfile.gettempdir()
        
        try:
            # Load data
            df = pd.read_excel(input_excel_path)
            
            # Simple analysis - just convert to GeoJSON
            if 'Koordinat GPS - Lintang' in df.columns and 'Koordinat GPS - Bujur' in df.columns:
                lat_col = 'Koordinat GPS - Lintang'
                lon_col = 'Koordinat GPS - Bujur'
            elif 'Latitude' in df.columns and 'Longitude' in df.columns:
                lat_col = 'Latitude'
                lon_col = 'Longitude'
            else:
                raise ValueError("Kolom koordinat tidak ditemukan")
            
            # Clean data
            df = df.dropna(subset=[lat_col, lon_col])
            
            # Create simple GeoJSON
            features = []
            for idx, row in df.iterrows():
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [row[lon_col], row[lat_col]]
                    },
                    "properties": {
                        "id": idx,
                        "type": "accident"
                    }
                }
                features.append(feature)
            
            geojson_data = {
                "type": "FeatureCollection",
                "features": features
            }
            
            # Save GeoJSON
            geojson_path = os.path.join(output_dir, 'accident_points.geojson')
            with open(geojson_path, 'w') as f:
                json.dump(geojson_data, f, indent=2)
            
            # Create simple HTML
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Accident Points</title>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
                <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
            </head>
            <body>
                <div id="map" style="height: 600px;"></div>
                <script>
                    var map = L.map('map').setView([{df[lat_col].mean()}, {df[lon_col].mean()}], 13);
                    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png').addTo(map);
                    
                    var geojsonData = {json.dumps(geojson_data)};
                    L.geoJSON(geojsonData, {{
                        pointToLayer: function (feature, latlng) {{
                            return L.circleMarker(latlng, {{radius: 5, fillColor: 'red', color: 'red'}});
                        }}
                    }}).addTo(map);
                </script>
            </body>
            </html>
            """
            
            html_path = os.path.join(output_dir, 'accident_map.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return geojson_path, html_path
            
        except Exception as e:
            raise Exception(f"Simple analysis failed: {str(e)}")
    
    analysis_function = simple_fallback_analysis
    analysis_module_available = True

# ========================================
# FLASK APP INITIALIZATION
# ========================================

app = Flask(__name__, template_folder='templates', static_folder='static')

# Konfigurasi Flask
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['DEBUG'] = True
app.config['PROPAGATE_EXCEPTIONS'] = True

# Buat folder yang diperlukan
required_folders = ['templates', 'static', 'uploads', 'cache']
for folder in required_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

def check_dependencies():
    """Check if all required Python packages are available"""
    required_packages = {
        'pandas': 'Data processing',
        'numpy': 'Numerical computing',
        'flask': 'Web framework'
    }
    
    optional_packages = {
        'geopandas': 'Spatial data processing',
        'sklearn': 'Machine learning (KDE)', 
        'shapely': 'Geometric operations',
        'scipy': 'Scientific computing',
        'osmnx': 'Road network analysis',
        'folium': 'Interactive maps'
    }

    missing = []
    available = []
    
    # Check required packages
    for package, description in required_packages.items():
        try:
            __import__(package)
            available.append(f" {package}: {description}")
        except ImportError:
            missing.append(f"âŒ {package}: {description} (REQUIRED)")
    
    # Check optional packages
    for package, description in optional_packages.items():
        try:
            mod = __import__(package)
            if hasattr(mod, '__version__'):
                version = getattr(mod, '__version__')
                available.append(f" {package}: {description} (v{version})")
            else:
                available.append(f" {package}: {description}")
        except ImportError:
            available.append(f"âš ï¸ {package}: {description} (optional)")

    return missing, available

@app.route('/health')
def health_check():
    """Enhanced health check with dependency validation"""
    try:
        missing, available = check_dependencies()
        status = "healthy" if len(missing) == 0 else "degraded"
        
        return jsonify({
            "status": status,
            "timestamp": datetime.datetime.now().isoformat(),
            "dependencies": {
                "available": available,
                "missing": missing
            },
            "analysis_module": "available" if analysis_module_available else "fallback_only",
            "message": "All core dependencies available" if not missing else f"Missing: {len(missing)} critical packages"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Health check failed: {str(e)}",
            "timestamp": datetime.datetime.now().isoformat()
        }), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """Enhanced analyze endpoint with comprehensive error handling"""
    temp_dir = None
    input_path = None
    
    try:
        print("Memulai analisis...")
        
        # ========================================
        # VALIDASI REQUEST
        # ========================================
        if 'file' not in request.files:
            return jsonify({"error": "Tidak ada file yang dikirim"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Tidak ada file yang dipilih"}), 400
        
        # Validasi ekstensi
        allowed_extensions = {'.xlsx', '.csv', '.xls'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({
                "error": f"Format file tidak didukung. Gunakan: {', '.join(allowed_extensions)}"
            }), 400
        
        # Get file size
        file_content = file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        if file_size_mb > 50:
            return jsonify({
                "error": f"File terlalu besar ({file_size_mb:.1f}MB). Maksimal 50MB."
            }), 400
        
        file.seek(0)  # Reset file pointer
        print(f"File valid: {file.filename} ({file_size_mb:.2f}MB)")
        
        # ========================================
        # SETUP DIREKTORI SEMENTARA
        # ========================================
        temp_dir = tempfile.mkdtemp(prefix='hotspot_analysis_')
        print(f"Temp directory: {temp_dir}")
        
        # Simpan file input
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        input_filename = f"{timestamp}-{os.path.basename(file.filename)}"
        input_path = os.path.join(temp_dir, input_filename)
        
        with open(input_path, 'wb') as f:
            f.write(file_content)
        
        print(f"File saved: {input_path}")
        
        # ========================================
        # VALIDASI AWAL FILE
        # ========================================
        print("Validasi awal file...")
        try:
            import pandas as pd
            if file_ext in ['.xlsx', '.xls']:
                df_test = pd.read_excel(input_path, nrows=5)
            elif file_ext == '.csv':
                df_test = pd.read_csv(input_path, nrows=5)
            
            print(f"File preview: {len(df_test)} rows, columns: {list(df_test.columns)}")
            
            # Cek kolom koordinat
            coord_columns = ['Koordinat GPS - Lintang', 'Koordinat GPS - Bujur', 'Latitude', 'Longitude']
            found_coords = [col for col in coord_columns if col in df_test.columns]
            
            if len(found_coords) < 2:
                return jsonify({
                    "error": "Kolom koordinat tidak ditemukan",
                    "found_columns": list(df_test.columns),
                    "expected_columns": coord_columns
                }), 400
                
        except Exception as validation_error:
            print(f"File validation error: {validation_error}")
            return jsonify({
                "error": f"File tidak dapat dibaca: {str(validation_error)}",
                "suggestion": "Pastikan file Excel/CSV tidak corrupt dan memiliki format yang benar"
            }), 400
        
        # ========================================
        # JALANKAN ANALISIS
        # ========================================
        print("Memulai analisis...")
        
        try:
            # Call analysis function dengan parameters yang minimal
            if analysis_function:
                result_geojson_path, result_html_path = analysis_function(
                    input_excel_path=input_path,
                    output_dir=temp_dir,
                    max_workers=1  # Single thread untuk stability
                )
            else:
                raise Exception("No analysis function available")
                
            print(" Analisis berhasil!")
            
        except Exception as analysis_error:
            print(f"Analisis error: {analysis_error}")
            traceback.print_exc()
            
            # Enhanced error classification
            error_str = str(analysis_error).lower()
            
            if "koordinat" in error_str or "coordinate" in error_str:
                return jsonify({
                    "error": "Masalah dengan data koordinat",
                    "details": str(analysis_error),
                    "suggestions": [
                        "Pastikan file memiliki kolom koordinat yang benar",
                        "Periksa format koordinat (gunakan angka desimal)",
                        "Pastikan koordinat tidak kosong atau bernilai 0"
                    ]
                }), 400
                
            elif "osmnx" in error_str or "network" in error_str:
                return jsonify({
                    "error": "Masalah dengan analisis jaringan jalan", 
                    "details": str(analysis_error),
                    "suggestions": [
                        "Server menggunakan metode fallback",
                        "Hasil tetap dapat diperoleh dengan metode sederhana"
                    ]
                }), 500
                
            elif "memory" in error_str or "ram" in error_str:
                return jsonify({
                    "error": "Server kehabisan memori",
                    "details": str(analysis_error),
                    "suggestions": [
                        "Coba dengan file yang lebih kecil",
                        "Kurangi jumlah data dalam file"
                    ]
                }), 500
                
            else:
                return jsonify({
                    "error": "Analisis gagal",
                    "details": str(analysis_error),
                    "suggestions": [
                        "Periksa format file dan data koordinat",
                        "Coba dengan file yang lebih kecil",
                        "Hubungi administrator jika masalah berlanjut"
                    ]
                }), 500
        
        # ========================================
        # VALIDASI HASIL
        # ========================================
        if not os.path.exists(result_geojson_path):
            return jsonify({"error": "File hasil GeoJSON tidak terbuat"}), 500
        
        if not os.path.exists(result_html_path):
            return jsonify({"error": "File hasil HTML tidak terbuat"}), 500
        
        print(f"Results created:")
        print(f" - GeoJSON: {result_geojson_path}")
        print(f" - HTML: {result_html_path}")
        
        # ========================================
        # BUAT FILE ZIP
        # ========================================
        print("Creating ZIP package...")
        memory_file = io.BytesIO()
        
        try:
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add main results
                zf.write(result_geojson_path, arcname=os.path.basename(result_geojson_path))
                zf.write(result_html_path, arcname=os.path.basename(result_html_path))
                
                # Add README
                readme_content = f"""# Hasil Analisis Hotspot Kecelakaan Lalu Lintas

## File yang Disertakan:
- {os.path.basename(result_geojson_path)}: Data hasil analisis dalam format GeoJSON
- {os.path.basename(result_html_path)}: Visualisasi peta interaktif

## Cara Menggunakan:
1. Buka file HTML di browser untuk melihat peta
2. Import file GeoJSON ke software GIS untuk analisis lanjutan

## Informasi Analisis:
- Tanggal: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- File Input: {file.filename}
- Metode: {"Advanced KDE" if "run_analysis" in str(analysis_function) else "Simple Point Analysis"}

## Catatan:
Untuk pertanyaan teknis, hubungi administrator sistem.
"""
                zf.writestr('README.txt', readme_content)
            
            memory_file.seek(0)
            zip_size_mb = len(memory_file.getvalue()) / (1024 * 1024)
            print(f" ZIP created: {zip_size_mb:.2f}MB")
            
        except Exception as zip_error:
            print(f"ZIP creation error: {zip_error}")
            return jsonify({"error": f"Gagal membuat file ZIP: {str(zip_error)}"}), 500
        
        # ========================================
        # KIRIM FILE HASIL
        # ========================================
        download_filename = f'analisis_hotspot_{timestamp}.zip'
        
        return send_file(
            memory_file,
            download_name=download_filename,
            as_attachment=True,
            mimetype='application/zip'
        )
        
    except Exception as e:
        print(f"General error: {e}")
        traceback.print_exc()
        
        return jsonify({
            "error": "Terjadi kesalahan pada server",
            "details": str(e),
            "timestamp": datetime.datetime.now().isoformat(),
            "suggestions": [
                "Periksa format file dan data",
                "Coba dengan file yang lebih kecil",
                "Hubungi administrator jika masalah berlanjut"
            ]
        }), 500
        
    finally:
        # Cleanup
        try:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"Temp directory cleaned: {temp_dir}")
        except Exception as cleanup_error:
            print(f"Cleanup warning: {cleanup_error}")

# ========================================
# ERROR HANDLERS
# ========================================

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({
        "error": "File terlalu besar",
        "message": "Ukuran file melebihi batas maksimal 50MB",
        "suggestions": [
            "Kurangi jumlah data dalam file",
            "Kompres file Excel/CSV",
            "Bagi data menjadi beberapa file kecil"
        ]
    }), 413

@app.errorhandler(500)
def internal_server_error(e):
    traceback.print_exc()
    return jsonify({
        "error": "Internal server error",
        "message": "Terjadi kesalahan pada server",
        "suggestions": [
            "Coba lagi dalam beberapa menit",
            "Pastikan file format sudah benar", 
            "Hubungi administrator jika masalah berlanjut"
        ]
    }), 500

@app.errorhandler(404)
def not_found_error(e):
    return jsonify({
        "error": "Endpoint tidak ditemukan",
        "message": "URL yang diminta tidak tersedia",
        "available_endpoints": [
            "/health - Status server",
            "/upload - Halaman upload", 
            "/analyze - Endpoint analisis"
        ]
    }), 404

@app.route('/test')
def test_dependencies():
    """Test endpoint untuk memeriksa dependencies"""
    missing, available = check_dependencies()
    
    # Test simple analysis
    can_analyze = True
    test_message = "System ready"
    
    try:
        import pandas as pd
        import tempfile
        import os
        
        # Create simple test
        test_dir = tempfile.mkdtemp()
        test_data = pd.DataFrame({
            'Koordinat GPS - Lintang': [-5.147665, -5.148665],
            'Koordinat GPS - Bujur': [119.432732, 119.433732]
        })
        test_file = os.path.join(test_dir, 'test.xlsx')
        test_data.to_excel(test_file, index=False)
        
        # Test analysis function
        if analysis_function:
            result_geojson, result_html = analysis_function(test_file, test_dir)
            test_message = "Analysis function working"
        else:
            can_analyze = False
            test_message = "No analysis function available"
            
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        
    except Exception as e:
        can_analyze = False
        test_message = f"Analysis test failed: {str(e)}"
    
    return jsonify({
        "timestamp": datetime.datetime.now().isoformat(),
        "dependencies": {
            "available": available,
            "missing": missing
        },
        "status": "ready" if (not missing and can_analyze) else "limited",
        "can_analyze": can_analyze,
        "test_message": test_message,
        "analysis_module": "available" if analysis_module_available else "fallback_only"
    })

if __name__ == '__main__':
    print("Starting Flask server...")
    print("=" * 60)
    
    # Check dependencies
    print("Checking dependencies...")
    missing, available = check_dependencies()
    
    print("\nAvailable packages:")
    for pkg in available:
        print(f"  {pkg}")
    
    if missing:
        print("\nMissing packages:")
        for pkg in missing:
            print(f"  {pkg}")
        
        # Check if critical packages are missing
        critical_missing = [m for m in missing if "REQUIRED" in m]
        if critical_missing:
            print(f"\nCritical packages missing: {len(critical_missing)}")
            print("Install with: pip install " + " ".join([
                m.split(":")[0].replace("", "") for m in critical_missing
            ]))
            sys.exit(1)
    
    print(f"\nAnalysis module: {'Available' if analysis_module_available else 'Fallback only'}")
    
    print("\nServer endpoints:")
    print("  - Main: http://127.0.0.1:5000")
    print("  - Upload: http://127.0.0.1:5000/upload")
    print("  - Health: http://127.0.0.1:5000/health")
    print("  - Test: http://127.0.0.1:5000/test")
    
    print("\n" + "=" * 60)
    print("Server ready for hotspot analysis!")
    
    # Run Flask app
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)