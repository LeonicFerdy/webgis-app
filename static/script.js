// Fungsi untuk memetakan nilai ke warna dengan gradasi yang lebih halus
function mapColor(value, vmin, vmax) {
    const normalized = Math.max(0, Math.min(1, (value - vmin) / (vmax - vmin)));
    // Gradasi warna dari biru (rendah) ke merah (tinggi) melalui kuning
    if (normalized < 0.33) {
        const ratio = normalized / 0.33;
        return `rgb(${Math.floor(0 + ratio * 255)}, ${Math.floor(0 + ratio * 255)}, 255)`;
    } else if (normalized < 0.66) {
        const ratio = (normalized - 0.33) / 0.33;
        return `rgb(255, 255, ${Math.floor(255 - ratio * 255)})`;
    } else {
        const ratio = (normalized - 0.66) / 0.34;
        return `rgb(255, ${Math.floor(255 - ratio * 255)}, 0)`;
    }
}

// Inisialisasi peta dengan koordinat Kabupaten Gowa yang lebih tepat
const map = L.map('map').setView([-5.168, 119.465], 11);

// Menambahkan beberapa pilihan basemap
const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
});

const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: '© Esri'
});

osmLayer.addTo(map);

// Layer groups untuk mengatur visibility
let kecelakaanLayer;
let hotspotLayer;
let roadsLayer;
let overlays = {};

// ---------- Tampilkan Titik Kecelakaan ---------- //
fetch('static/lakalantas.geojson') //Loading data kecelakaan secara asynchronous
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Data kecelakaan loaded:', data.features.length, 'features');
        
        kecelakaanLayer = L.geoJSON(data, {
            pointToLayer: (feature, latlng) => {
                let level = feature.properties["Tingkat Kecelakaan"];
                let color = "green"; // Default sekarang hijau (kecelakaan ringan)
                let radius = 5;
                
                // Normalisasi dan validasi data tingkat kecelakaan
                if (level) {
                    level = level.toString().trim().toLowerCase();
                    
                    if (level === "ringan" || level === "rendah" || level === "kecil") {
                        level = "Ringan";
                        color = "green";
                        radius = 5;
                    } else if (level === "sedang" || level === "menengah" || level === "moderate") {
                        level = "Sedang";
                        color = "orange";
                        radius = 7;
                    } else if (level === "berat" || level === "tinggi" || level === "parah" || level === "severe") {
                        level = "Berat";
                        color = "red";
                        radius = 9;
                    } else {
                        // Jika tingkat kecelakaan tidak dikenali, set sebagai ringan
                        level = "Ringan";
                        color = "green";
                        radius = 5;
                    }
                } else {
                    // Jika tidak ada data tingkat kecelakaan, set sebagai ringan
                    level = "Ringan";
                    color = "green";
                    radius = 5;
                }
                
                // Update property tingkat kecelakaan yang sudah dinormalisasi
                feature.properties["Tingkat Kecelakaan"] = level;

                return L.circleMarker(latlng, {
                    radius: radius,
                    fillColor: color,
                    color: "black",
                    weight: 1,
                    fillOpacity: 0.8
                }).bindPopup(`
                    <div style="font-family: Arial, sans-serif;">
                        <h4 style="margin: 0 0 10px 0; color: #333;">Detail Kecelakaan</h4>
                        <p><strong>ID:</strong> ${feature.properties["ID"] || 'N/A'}</p>
                        <p><strong>Tanggal:</strong> ${feature.properties["Tanggal Kejadian"] || 'N/A'}</p>
                        <p><strong>Tingkat:</strong> <span style="color: ${color}; font-weight: bold;">${level}</span></p>
                        <p><strong>Lokasi:</strong> ${latlng.lat.toFixed(6)}, ${latlng.lng.toFixed(6)}</p>
                    </div>
                `);
            }
        });

        overlays["Titik Kecelakaan"] = kecelakaanLayer;
        updateLayerControl();
        
        // Tambahkan layer ke peta secara default
        kecelakaanLayer.addTo(map);
        
        // Update statistik setelah data dimuat
        updateStatistics();
    })
    .catch(error => {
        console.error('Error loading kecelakaan data:', error);
        showNotification('Gagal memuat data kecelakaan', 'error');
    });

// ---------- Tampilkan Hotspot dan Jalan ---------- //
fetch('static/hotspot_analysis_with_roads_kabupaten_gowa.geojson')
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Data hotspot loaded:', data.features.length, 'features');
        
        // Pisahkan hotspot dan roads
        const hotspotFeatures = [];
        const roadFeatures = [];
        
        // TAMBAHAN: Pisahkan berdasarkan kategori untuk filter
        const categorizedFeatures = {
            hotspot_all: [],
            hotspot_2021: [],
            hotspot_2022: [],
            hotspot_2023: [],
            hotspot_sedang: [],
            hotspot_berat: [],
            roads_all: [],
            roads_2021: [],
            roads_2022: [],
            roads_2023: [],
            roads_sedang: [],
            roads_berat: []
        };
        
        data.features.forEach(feature => {
            if (feature.geometry.type === 'Point') {
                hotspotFeatures.push(feature);
                
                // Kategorikan untuk filter
                if (feature.properties.category === 'hotspot_all_data') {
                    categorizedFeatures.hotspot_all.push(feature);
                } else if (feature.properties.category === 'hotspot_yearly' && feature.properties.year == 2021) {
                    categorizedFeatures.hotspot_2021.push(feature);
                } else if (feature.properties.category === 'hotspot_yearly' && feature.properties.year == 2022) {
                    categorizedFeatures.hotspot_2022.push(feature);
                } else if (feature.properties.category === 'hotspot_yearly' && feature.properties.year == 2023) {
                    categorizedFeatures.hotspot_2023.push(feature);
                } else if (feature.properties.category === 'hotspot_severity' && feature.properties.severity === 'Sedang') {
                    categorizedFeatures.hotspot_sedang.push(feature);
                } else if (feature.properties.category === 'hotspot_severity' && feature.properties.severity === 'Berat') {
                    categorizedFeatures.hotspot_berat.push(feature);
                }
                
            } else if (feature.geometry.type === 'LineString') {
                roadFeatures.push(feature);
                
                // Kategorikan roads untuk filter
                if (feature.properties.category === 'road_segment_all_data') {
                    categorizedFeatures.roads_all.push(feature);
                } else if (feature.properties.category === 'road_segment_yearly' && feature.properties.year == 2021) {
                    categorizedFeatures.roads_2021.push(feature);
                } else if (feature.properties.category === 'road_segment_yearly' && feature.properties.year == 2022) {
                    categorizedFeatures.roads_2022.push(feature);
                } else if (feature.properties.category === 'road_segment_yearly' && feature.properties.year == 2023) {
                    categorizedFeatures.roads_2023.push(feature);
                } else if (feature.properties.category === 'road_segment_severity' && feature.properties.severity === 'Sedang') {
                    categorizedFeatures.roads_sedang.push(feature);
                } else if (feature.properties.category === 'road_segment_severity' && feature.properties.severity === 'Berat') {
                    categorizedFeatures.roads_berat.push(feature);
                }
            }
        });

        console.log('Hotspot points:', hotspotFeatures.length);
        console.log('Road segments:', roadFeatures.length);

        // Layer untuk hotspot (semua data) - DEFAULT
        if (categorizedFeatures.hotspot_all.length > 0) {
            hotspotLayer = L.geoJSON({type: "FeatureCollection", features: categorizedFeatures.hotspot_all}, {
                pointToLayer: (feature, latlng) => {
                    const density = feature.properties.density || 0;
                    const color = feature.properties.color || "#999";
                    
                    return L.circleMarker(latlng, {
                        radius: Math.max(4, Math.min(12, density * 10)), // Radius berdasarkan kepadatan
                        fillColor: color,
                        color: "#000",
                        weight: 1,
                        fillOpacity: 0.7
                    }).bindPopup(`
                        <div style="font-family: Arial, sans-serif;">
                            <h4 style="margin: 0 0 10px 0; color: #333;">Hotspot Kecelakaan</h4>
                            <p><strong>Kepadatan:</strong> ${density.toFixed(4)}</p>
                            <p><strong>Koordinat:</strong> ${latlng.lat.toFixed(6)}, ${latlng.lng.toFixed(6)}</p>
                            <p><strong>Status:</strong> <span style="color: red; font-weight: bold;">Area Rawan</span></p>
                        </div>
                    `);
                }
            });
            overlays["Hotspot Area (Semua Data)"] = hotspotLayer;
            hotspotLayer.addTo(map); 
        }
        // Layer untuk roads (semua data) - DEFAULT
        if (categorizedFeatures.roads_all.length > 0) {
            roadsLayer = L.geoJSON({type: "FeatureCollection", features: categorizedFeatures.roads_all}, {
                style: feature => {
                    const color = feature.properties.color || "#666";
                    return {
                        color: color,
                        weight: 3,
                        opacity: 0.8
                    };
                }
            }).bindPopup("Ruas Jalan dalam Area Hotspot");
            overlays["Jalan Rawan (Semua Data)"] = roadsLayer;
            roadsLayer.addTo(map); 
        }

        // TAMBAHKAN FILTER LAYER - 2021
        if (categorizedFeatures.hotspot_2021.length > 0) {
            const hotspot2021Layer = L.geoJSON({type: "FeatureCollection", features: categorizedFeatures.hotspot_2021}, {
                pointToLayer: createHotspotMarker
            });
            overlays["🔹 Filter: Hotspot 2021"] = hotspot2021Layer;
        }
        
        // TAMBAHKAN JALAN RAWAN 2021
        if (categorizedFeatures.roads_2021.length > 0) {
            const roads2021Layer = L.geoJSON({type: "FeatureCollection", features: categorizedFeatures.roads_2021}, {
                style: createRoadStyle
            }).bindPopup("Jalan Rawan 2021 (150m dari hotspot)");
            overlays["🔹 Jalan Rawan 2021"] = roads2021Layer;
        }

        // TAMBAHKAN FILTER LAYER - 2022
        if (categorizedFeatures.hotspot_2022.length > 0) {
            const hotspot2022Layer = L.geoJSON({type: "FeatureCollection", features: categorizedFeatures.hotspot_2022}, {
                pointToLayer: createHotspotMarker
            });
            overlays["🔹 Filter: Hotspot 2022"] = hotspot2022Layer;
        }
        
        // TAMBAHKAN JALAN RAWAN 2022
        if (categorizedFeatures.roads_2022.length > 0) {
            const roads2022Layer = L.geoJSON({type: "FeatureCollection", features: categorizedFeatures.roads_2022}, {
                style: createRoadStyle
            }).bindPopup("Jalan Rawan 2022 (150m dari hotspot)");
            overlays["🔹 Jalan Rawan 2022"] = roads2022Layer;
        }

        // TAMBAHKAN FILTER LAYER - 2023
        if (categorizedFeatures.hotspot_2023.length > 0) {
            const hotspot2023Layer = L.geoJSON({type: "FeatureCollection", features: categorizedFeatures.hotspot_2023}, {
                pointToLayer: createHotspotMarker
            });
            overlays["🔹 Filter: Hotspot 2023"] = hotspot2023Layer;
        }
        
        // TAMBAHKAN JALAN RAWAN 2023
        if (categorizedFeatures.roads_2023.length > 0) {
            const roads2023Layer = L.geoJSON({type: "FeatureCollection", features: categorizedFeatures.roads_2023}, {
                style: createRoadStyle
            }).bindPopup("Jalan Rawan 2023 (150m dari hotspot)");
            overlays["🔹 Jalan Rawan 2023"] = roads2023Layer;
        }

        // TAMBAHKAN FILTER LAYER - SEDANG
        if (categorizedFeatures.hotspot_sedang.length > 0) {
            const hotspotSedangLayer = L.geoJSON({type: "FeatureCollection", features: categorizedFeatures.hotspot_sedang}, {
                pointToLayer: createHotspotMarker
            });
            overlays["🔸 Filter: Hotspot Sedang"] = hotspotSedangLayer;
        }
        
        // TAMBAHKAN JALAN RAWAN SEDANG
        if (categorizedFeatures.roads_sedang.length > 0) {
            const roadsSedangLayer = L.geoJSON({type: "FeatureCollection", features: categorizedFeatures.roads_sedang}, {
                style: createRoadStyle
            }).bindPopup("Jalan Rawan Sedang (150m dari hotspot)");
            overlays["🔸 Jalan Rawan Sedang"] = roadsSedangLayer;
        }

        // TAMBAHKAN FILTER LAYER - BERAT
        if (categorizedFeatures.hotspot_berat.length > 0) {
            const hotspotBeratLayer = L.geoJSON({type: "FeatureCollection", features: categorizedFeatures.hotspot_berat}, {
                pointToLayer: createHotspotMarker
            });
            overlays["🔴 Filter: Hotspot Berat"] = hotspotBeratLayer;
        }
        
        // TAMBAHKAN JALAN RAWAN BERAT
        if (categorizedFeatures.roads_berat.length > 0) {
            const roadsBeratLayer = L.geoJSON({type: "FeatureCollection", features: categorizedFeatures.roads_berat}, {
                style: createRoadStyle
            }).bindPopup("Jalan Rawan Berat (150m dari hotspot)");
            overlays["🔴 Jalan Rawan Berat"] = roadsBeratLayer;
        }

        updateLayerControl();
        
        // Notifikasi 
        if (hotspotFeatures.length > 0) {
            showNotification('Data hotspot berhasil dimuat!', 'success');
        }
    })
    .catch(error => {
        console.error('Error loading hotspot data:', error);
        showNotification('Gagal memuat data hotspot: ' + error.message, 'error');
    });

// Fungsi helper untuk membuat hotspot marker
function createHotspotMarker(feature, latlng) {
    const density = feature.properties.density || 0;
    const color = feature.properties.color || "#999";
    
    return L.circleMarker(latlng, {
        radius: Math.max(4, Math.min(12, density * 10)),
        fillColor: color,
        color: "#000",
        weight: 1,
        fillOpacity: 0.7
    }).bindPopup(`
        <div style="font-family: Arial, sans-serif;">
            <h4 style="margin: 0 0 10px 0; color: #333;">Hotspot Kecelakaan</h4>
            <p><strong>Kepadatan:</strong> ${density.toFixed(4)}</p>
            <p><strong>Koordinat:</strong> ${latlng.lat.toFixed(6)}, ${latlng.lng.toFixed(6)}</p>
            <p><strong>Status:</strong> <span style="color: red; font-weight: bold;">Area Rawan</span></p>
        </div>
    `);
}

// Fungsi helper untuk membuat road style
function createRoadStyle(feature) {
    const color = feature.properties.color || "#666";
    return {
        color: color,
        weight: 3,
        opacity: 0.8
    };
}

// ---------- Layer Control ---------- //
let layerControl;

function updateLayerControl() {
    if (layerControl) {
        map.removeControl(layerControl);
    }
    
    const baseLayers = {
        "OpenStreetMap": osmLayer,
        "Satelit": satelliteLayer
    };
    
    layerControl = L.control.layers(baseLayers, overlays, { 
        collapsed: false,
        position: 'topright'
    }).addTo(map);

    // Event listeners untuk checkbox filter
    setupFilterListeners();
}

// Setup event listeners untuk filter checkbox
function setupFilterListeners() {
    const filterIds = ['filter-ringan', 'filter-sedang', 'filter-berat'];
    const filterLevels = ['Ringan', 'Sedang', 'Berat'];
    // mencegah bug akibat multiple listeners
    filterIds.forEach((id, index) => {
        const checkbox = document.getElementById(id);
        if (checkbox) {
            // Remove existing listeners untuk mencegah duplikasi
            checkbox.removeEventListener('change', checkbox._toggleHandler);
            
            // Add new listener
            checkbox._toggleHandler = function() {
                toggleLayerVisibility(filterLevels[index], this.checked);
            };
            checkbox.addEventListener('change', checkbox._toggleHandler);
        }
    });
}

// Fungsi untuk menampilkan/menghilangkan layer berdasarkan filter
function toggleLayerVisibility(level, isVisible) {
    if (!kecelakaanLayer) return;
    
    kecelakaanLayer.eachLayer(layer => {
        const featureLevel = layer.feature.properties["Tingkat Kecelakaan"];
        if (featureLevel === level) {
            if (isVisible) {
                if (!map.hasLayer(layer)) {
                    map.addLayer(layer);
                }
            } else {
                if (map.hasLayer(layer)) {
                    map.removeLayer(layer);
                }
            }
        }
    });
    
    // Update statistik setelah filter berubah
    updateStatistics();
}

// Fungsi untuk menampilkan notifikasi (menggunakan CSS class)
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

// ---------- Fitur Tambahan ---------- //

// Kontrol zoom ke hotspot
function zoomToHotspots() {
    if (hotspotLayer && hotspotLayer.getLayers().length > 0) {
        const group = new L.featureGroup(hotspotLayer.getLayers());
        map.fitBounds(group.getBounds(), {padding: [20, 20]});
        showNotification('Zoom ke area hotspot', 'info');
    } else {
        showNotification('Data hotspot tidak tersedia', 'warning');
    }
}

// Kontrol untuk menampilkan statistik
function updateStatistics() {
    setTimeout(() => {
        if (kecelakaanLayer) {
            const stats = {
                total: 0,
                ringan: 0,
                sedang: 0,
                berat: 0,
                visible: 0
            };
            
            kecelakaanLayer.eachLayer(layer => {
                const level = layer.feature.properties["Tingkat Kecelakaan"];
                stats.total++;
                if (map.hasLayer(layer)) {
                    stats.visible++;
                }
                if (level === "Ringan") stats.ringan++;
                else if (level === "Sedang") stats.sedang++;
                else if (level === "Berat") stats.berat++;
            });
            
            const statsElement = document.getElementById('statistics');
            if (statsElement) {
                statsElement.innerHTML = `
                    <h3>Ringkasan Data Kecelakaan</h3>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-number">${stats.total}</div>
                            <div class="stat-label">Total Kecelakaan</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number" style="color: green;">${stats.ringan}</div>
                            <div class="stat-label">Kecelakaan Ringan</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number" style="color: orange;">${stats.sedang}</div>
                            <div class="stat-label">Kecelakaan Sedang</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-number" style="color: red;">${stats.berat}</div>
                            <div class="stat-label">Kecelakaan Berat</div>
                        </div>
                    </div>
                    <p style="margin-top: 15px; color: #666; font-style: italic;">
                        Data dianalisis menggunakan algoritma Kernel Density Estimation (KDE)<br>
                        Saat ini menampilkan: ${stats.visible} dari ${stats.total} data kecelakaan
                    </p>
                `;
            }
        }
    }, 1000);
}

// Event listener untuk update statistik setelah layer berubah
map.on('layeradd', function() {
    updateStatistics();
});

map.on('layerremove', function() {
    updateStatistics();
});

// Console log untuk debugging
console.log('Script loaded successfully');