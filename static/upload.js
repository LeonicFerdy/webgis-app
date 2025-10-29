// Upload Page JavaScript - Enhanced Version with Better Error Handling
// =============================================================================

// Parameter analisis yang bisa Anda tampilkan jika perlu
const HOTSPOT_PARAMS = {
    bandwidth: 0.008,
    zscoreThreshold: 1.96,
    roadBuffer: 150,
};

// Variabel Global untuk menyimpan state
let uploadedData = null;
let currentFile = null;
let isProcessing = false;

// DOM Elements yang sering digunakan
let uploadArea, fileInput, previewSection, processSection, progressArea, resultsArea;

// Fungsi ini akan berjalan setelah seluruh halaman dimuat
document.addEventListener('DOMContentLoaded', function() {
    console.log('Halaman Upload siap.');

    // Inisialisasi semua elemen dari halaman HTML
    uploadArea = document.getElementById('uploadArea');
    fileInput = document.getElementById('fileInput');
    previewSection = document.getElementById('previewSection');
    processSection = document.getElementById('processSection');
    progressArea = document.getElementById('progressArea');
    resultsArea = document.getElementById('resultsArea');

    // Menyiapkan semua event listener
    setupEventListeners();

    // Check server health on startup (dengan graceful handling)
    checkServerHealth();

    showNotification('Sistem siap untuk menerima data unggahan!', 'success');
});

/**
 * Check if the Flask server is running and healthy
 */
async function checkServerHealth() {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 detik timeout

        const response = await fetch('/health', { 
            method: 'GET',
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (response.ok) {
            const data = await response.json();
            console.log('Server health check passed:', data);
            
            // Check for OSMnx version and potential compatibility issues
            if (data.osmnx_version) {
                const version = data.osmnx_version;
                showNotification(`Server siap! OSMnx v${version}`, 'success');
                
                // Warn about potential version issues
                const majorVersion = parseInt(version.split('.')[0]);
                if (majorVersion >= 1) {
                    console.warn('OSMnx v1.x detected - checking for API compatibility...');
                }
            } else {
                showNotification('Server siap! OSMnx terdeteksi', 'success');
            }
        } else {
            throw new Error(`Server health check failed: ${response.status}`);
        }
    } catch (error) {
        console.warn('Server health check failed:', error);
        
        // Handle specific error types gracefully
        if (error.name === 'AbortError') {
            console.warn('Health check timeout - server might be slow');
            showNotification('⚠️ Server lambat merespons, tapi sistem tetap dapat digunakan.', 'warning');
        } else if (error.message.includes('404') || error.message.includes('Failed to fetch')) {
            console.warn('Health endpoint not available - this is normal for some setups');
            showNotification('Sistem siap! (Health check endpoint tidak tersedia)', 'info');
        } else {
            showNotification('⚠️ Peringatan: Server mungkin belum siap atau ada masalah konfigurasi. Coba proses file untuk memastikan.', 'warning');
        }
    }
}

/**
 * Menyiapkan semua event listener untuk interaksi pengguna.
 */
function setupEventListeners() {
    if (!uploadArea || !fileInput) {
        console.error('Elemen penting untuk upload tidak ditemukan!');
        return;
    }

    // Event listener untuk input file
    fileInput.addEventListener('change', handleFileSelect);

    // Event listener untuk area drag and drop
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Event listener untuk tombol-tombol
    document.getElementById('processBtn')?.addEventListener('click', startProcessing);
    document.getElementById('resetBtn')?.addEventListener('click', resetUpload);
    document.getElementById('downloadSample')?.addEventListener('click', generateSampleFile);
}

// =============================================================================
// BAGIAN UTAMA: FUNGSI UNTUK MEMPROSES DAN MENGIRIM FILE
// =============================================================================

/**
 * Memulai proses analisis dengan mengirim file ke server backend.
 */
async function startProcessing() {
    if (!currentFile) {
        showNotification('Tidak ada data untuk diproses!', 'error');
        return;
    }

    if (isProcessing) {
        showNotification('Proses sedang berjalan, harap tunggu...', 'warning');
        return;
    }

    console.log('Mengirim file ke server untuk analisis...');
    showNotification('Memulai analisis hotspot kecelakaan...', 'info');

    // Set processing state
    isProcessing = true;

    // Tampilkan area progress dan atur ke state awal
    if (progressArea) {
        progressArea.style.display = 'block';
        progressArea.classList.add('fade-in');
        updateProgressBar('Memulai analisis...', 'Mengunggah file ke server...', 10);
    }
    
    // Nonaktifkan tombol agar tidak diklik dua kali
    const processBtn = document.getElementById('processBtn');
    if (processBtn) {
        processBtn.disabled = true;
        processBtn.textContent = 'Memproses...';
    }

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        updateProgressBar('Mengunggah...', 'File sedang dikirim ke server...', 20);

        // Kirim permintaan POST ke endpoint /analyze di server Flask dengan timeout yang lebih panjang
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 menit timeout

        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        updateProgressBar('Menganalisis...', 'Server sedang melakukan analisis hotspot...', 50);

        if (!response.ok) {
            let errorMessage = 'Terjadi kesalahan pada server';
            let errorDetails = '';
            
            try {
                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    const errorData = await response.json();
                    errorMessage = errorData.error || errorMessage;
                    errorDetails = errorData.details || '';
                } else {
                    // If response is not JSON, try to get text
                    const errorText = await response.text();
                    errorMessage = `Server Error ${response.status}: ${response.statusText}`;
                    errorDetails = errorText;
                }
            } catch (parseError) {
                console.warn('Could not parse error response:', parseError);
                errorMessage = `Server Error ${response.status}: ${response.statusText}`;
            }
            
            // Provide specific error handling for common issues
            if (errorDetails.includes("has no attribute 'config'") || 
                errorDetails.includes("has no attribute 'settings'") ||
                errorMessage.includes("has no attribute 'config'")) {
                errorMessage = 'Masalah kompatibilitas OSMnx: Versi OSMnx di server tidak kompatibel. Silakan hubungi administrator untuk update configurasi server.';
                showOSMnxCompatibilityError();
            } else if (errorDetails.includes('osmnx') || errorMessage.includes('osmnx')) {
                errorMessage = 'Masalah dengan library OSMnx di server. Pastikan server dikonfigurasi dengan benar.';
            } else if (errorDetails.includes('timeout') || errorMessage.includes('timeout')) {
                errorMessage = 'Proses analisis memakan waktu terlalu lama. Coba dengan data yang lebih kecil atau area yang lebih spesifik.';
            } else if (errorDetails.includes('memory') || errorMessage.includes('memory')) {
                errorMessage = 'Server kehabisan memori. Coba dengan dataset yang lebih kecil.';
            } else if (errorDetails.includes('network') || errorMessage.includes('network')) {
                errorMessage = 'Masalah koneksi jaringan saat mengunduh data peta. Periksa koneksi internet server.';
            } else if (response.status === 404) {
                errorMessage = 'Endpoint analisis tidak ditemukan. Pastikan server Flask berjalan dengan benar.';
            } else if (response.status === 500) {
                errorMessage = 'Error internal server. Periksa log server untuk detail lebih lanjut.';
            }
            
            throw new Error(errorMessage);
        }

        updateProgressBar('Memproses hasil...', 'Menyiapkan file hasil analisis...', 80);

        // Jika berhasil, browser akan otomatis mengunduh file ZIP
        const blob = await response.blob();
        
        if (blob.size === 0) {
            throw new Error('File hasil analisis kosong. Periksa data input Anda.');
        }

        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;

        // Extract filename from response headers
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = `accident_analysis_${new Date().toISOString().slice(0,10)}.zip`; // Default filename
        
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
            if (filenameMatch && filenameMatch.length > 1) {
                filename = filenameMatch[1].replace(/['"]/g, '');
            }
        }
        
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();

        updateProgressBar('Selesai!', 'Analisis berhasil, file ZIP sudah diunduh.', 100);
        showNotification('Analisis hotspot selesai! File ZIP berhasil diunduh.', 'success');
        
        showResultsSummaryAfterDownload(filename);

    } catch (error) {
        console.error('Proses gagal:', error);
        
        let userFriendlyMessage = error.message;
        
        // Handle AbortError (timeout)
        if (error.name === 'AbortError') {
            userFriendlyMessage = 'Proses analisis dibatalkan karena timeout (lebih dari 5 menit). Coba dengan dataset yang lebih kecil.';
        }
        
        // Handle network errors
        if (error.message.includes('Failed to fetch') || error.name === 'TypeError') {
            userFriendlyMessage = 'Gagal terhubung ke server. Pastikan server Flask berjalan di localhost:5000.';
        }
        
        showNotification(`❌ Analisis gagal: ${userFriendlyMessage}`, 'error');
        
        if (progressArea) {
            progressArea.style.display = 'none';
        }
        
        // Show error details in console for debugging
        console.error('Detailed error information:', {
            message: error.message,
            stack: error.stack,
            fileName: currentFile?.name,
            fileSize: currentFile?.size,
            errorName: error.name
        });
        
    } finally {
        // Reset processing state
        isProcessing = false;
        
        // Aktifkan kembali tombol setelah proses selesai
        if (processBtn) {
            processBtn.disabled = false;
            processBtn.textContent = 'Mulai Analisis Hotspot';
        }
    }
}

/**
 * Show specific error message and solutions for OSMnx compatibility issues
 */
function showOSMnxCompatibilityError() {
    const errorDetailsHTML = `
        <div class="error-details" style="margin: 20px 0; padding: 20px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px;">
            <h4 style="color: #856404; margin-bottom: 15px;">🔧 Masalah Kompatibilitas OSMnx</h4>
            <p style="color: #856404; margin-bottom: 15px;">
                Server menggunakan versi OSMnx yang tidak kompatibel. Ini adalah masalah umum dengan versi OSMnx 1.x yang memiliki perubahan API.
            </p>
            <h5 style="color: #856404; margin-bottom: 10px;">Solusi untuk Administrator:</h5>
            <ol style="color: #856404; margin-left: 20px;">
                <li>Update kode server untuk menggunakan <code>ox.settings</code> bukan <code>ox.config</code></li>
                <li>Atau downgrade OSMnx ke versi 0.x: <code>pip install osmnx==0.16.2</code></li>
                <li>Periksa dokumentasi OSMnx terbaru untuk perubahan API</li>
            </ol>
            <p style="color: #856404; margin-top: 15px; font-style: italic;">
                Hubungi administrator sistem untuk memperbaiki masalah ini.
            </p>
        </div>
    `;
    
    // Add error details to the page if there's a suitable container
    const errorContainer = document.getElementById('errorDetails') || document.createElement('div');
    errorContainer.id = 'errorDetails';
    errorContainer.innerHTML = errorDetailsHTML;
    
    // Insert after progress area or at the end of process section
    const insertAfter = progressArea || processSection;
    if (insertAfter && insertAfter.parentNode && !document.getElementById('errorDetails')) {
        insertAfter.parentNode.insertBefore(errorContainer, insertAfter.nextSibling);
    }
}

// =============================================================================
// BAGIAN FILE HANDLING: FUNGSI-FUNGSI UNTUK MENANGANI FILE
// =============================================================================

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        currentFile = file;
        processFile(file);
    }
}

// Handle drag over
function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('dragover');
}

// Handle drag leave
function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
}

// Handle drop
function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        currentFile = files[0];
        processFile(files[0]);
    }
}

// Process uploaded file (parsing dan validasi di browser)
async function processFile(file) {
    console.log('Memproses file:', file.name);
    showNotification('Memvalidasi file...', 'info');
    
    try {
        const fileExtension = file.name.toLowerCase().split('.').pop();
        if (!['xlsx', 'csv', 'geojson'].includes(fileExtension)) {
            throw new Error('Format file tidak didukung. Gunakan .xlsx, .csv, atau .geojson');
        }
        
        // Check file size (limit to 50MB)
        const maxSize = 50 * 1024 * 1024; // 50MB
        if (file.size > maxSize) {
            throw new Error('File terlalu besar. Maksimal 50MB.');
        }
        
        let data;
        if (fileExtension === 'xlsx') {
            data = await parseExcelFile(file);
        } else if (fileExtension === 'csv') {
            data = await parseCSVFile(file);
        } else if (fileExtension === 'geojson') {
            data = await parseGeoJSONFile(file);
        }
        
        if (data && data.length > 0) {
            uploadedData = data;
            showPreview(file, data);
            showNotification(`✅ File berhasil divalidasi! ${data.length} data kecelakaan siap dianalisis.`, 'success');
        } else {
            throw new Error('Tidak ada data valid yang ditemukan di dalam file');
        }
        
    } catch (error) {
        console.error('Error memproses file:', error);
        showNotification(`❌ Error: ${error.message}`, 'error');
        resetUpload();
    }
}

// Parse Excel file - DIPERBAIKI
async function parseExcelFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const data = new Uint8Array(e.target.result);
                const workbook = XLSX.read(data, { type: 'array' });
                const sheetName = workbook.SheetNames[0];
                const worksheet = workbook.Sheets[sheetName];
                const jsonData = XLSX.utils.sheet_to_json(worksheet);
                
                // Debug: tampilkan semua kolom yang tersedia
                if (jsonData.length > 0) {
                    console.log('Kolom yang tersedia:', Object.keys(jsonData[0]));
                }
                
                resolve(processRawData(jsonData));
            } catch (error) {
                reject(new Error('Gagal membaca file Excel: ' + error.message));
            }
        };
        reader.onerror = () => reject(new Error('Gagal membaca file'));
        reader.readAsArrayBuffer(file);
    });
}

// Parse CSV file - DIPERBAIKI dengan fallback manual parsing
async function parseCSVFile(file) {
    return new Promise((resolve, reject) => {
        // Coba menggunakan PapaParse jika tersedia
        if (typeof Papa !== 'undefined' && Papa.parse) {
            Papa.parse(file, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: function(results) {
                    if (results.errors.length > 0) console.warn('Peringatan saat parsing CSV:', results.errors);
                    resolve(processRawData(results.data));
                },
                error: (error) => reject(new Error('Gagal membaca file CSV: ' + error.message))
            });
        } else {
            // Fallback: manual CSV parsing
            console.warn('PapaParse tidak tersedia, menggunakan manual parsing');
            const reader = new FileReader();
            reader.onload = function(e) {
                try {
                    const csvText = e.target.result;
                    const lines = csvText.split('\n');
                    const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
                    
                    const data = [];
                    for (let i = 1; i < lines.length; i++) {
                        const line = lines[i].trim();
                        if (line) {
                            const values = line.split(',').map(v => v.trim().replace(/"/g, ''));
                            const row = {};
                            headers.forEach((header, index) => {
                                row[header] = values[index] || '';
                            });
                            data.push(row);
                        }
                    }
                    
                    console.log('Manual CSV parsing berhasil, kolom:', headers);
                    resolve(processRawData(data));
                } catch (error) {
                    reject(new Error('Gagal parsing CSV manual: ' + error.message));
                }
            };
            reader.onerror = () => reject(new Error('Gagal membaca file'));
            reader.readAsText(file);
        }
    });
}

// Parse GeoJSON file
async function parseGeoJSONFile(file) {
     return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const geojsonData = JSON.parse(e.target.result);
                if (!geojsonData.features || !Array.isArray(geojsonData.features)) {
                    throw new Error('Format GeoJSON tidak valid');
                }
                const processed = geojsonData.features.map(f => ({
                    latitude: f.geometry.coordinates[1],
                    longitude: f.geometry.coordinates[0],
                    tanggal: f.properties['Tanggal Kejadian'] || f.properties.tanggal,
                    tingkat: f.properties['Tingkat Kecelakaan'] || f.properties.tingkat,
                    id: f.properties.ID || f.properties.id
                }));
                resolve(processed);
            } catch (error) {
                reject(new Error('Gagal membaca file GeoJSON: ' + error.message));
            }
        };
        reader.onerror = () => reject(new Error('Gagal membaca file'));
        reader.readAsText(file);
    });
}

// Proses data mentah dari format apapun - DIPERBAIKI dengan pencocokan yang lebih fleksibel
function processRawData(rawData) {
    if (!rawData || rawData.length === 0) {
        throw new Error('Data kosong atau tidak valid');
    }

    const columns = Object.keys(rawData[0] || {});
    console.log('Semua kolom tersedia:', columns);
    
    // Fungsi pencarian kolom yang lebih fleksibel
    const findColumn = (columns, potentialNames) => {
        // Normalisasi nama kolom untuk pencarian
        const normalizeColumnName = (name) => {
            return name.toLowerCase()
                .replace(/[\s_-]/g, '')
                .replace(/koordinat/g, '')
                .replace(/gps/g, '')
                .trim();
        };

        for (const potentialName of potentialNames) {
            const normalizedTarget = normalizeColumnName(potentialName);
            
            for (const col of columns) {
                const normalizedCol = normalizeColumnName(col);
                if (normalizedCol.includes(normalizedTarget) || normalizedTarget.includes(normalizedCol)) {
                    console.log(`Kolom ditemukan: '${col}' cocok dengan '${potentialName}'`);
                    return col;
                }
            }
        }
        return null;
    };
    
    // Cari kolom latitude dengan variasi nama yang lebih lengkap
    const latCol = findColumn(columns, [
        'latitude', 'lat', 'y', 'lintang', 'lintang', 'koordinatlintang', 
        'koordinat_lintang', 'koordinat-lintang', 'koordinatgpslintang',
        'koordinat_gps_lintang', 'koordinat-gps-lintang'
    ]);
    
    // Cari kolom longitude dengan variasi nama yang lebih lengkap  
    const lonCol = findColumn(columns, [
        'longitude', 'lon', 'lng', 'x', 'bujur', 'bujur', 'koordinatbujur',
        'koordinat_bujur', 'koordinat-bujur', 'koordinatgpsbujur',
        'koordinat_gps_bujur', 'koordinat-gps-bujur'
    ]);
    
    // Cari kolom tanggal
    const dateCol = findColumn(columns, [
        'tanggalkejadian', 'tanggal_kejadian', 'tanggal-kejadian', 
        'tanggal', 'date', 'waktu', 'tgl'
    ]);
    
    // Cari kolom tingkat kecelakaan
    const levelCol = findColumn(columns, [
        'tingkatkecelakaan', 'tingkat_kecelakaan', 'tingkat-kecelakaan',
        'tingkat', 'level', 'severity', 'keparahan'
    ]);

    console.log('Hasil pencarian kolom:');
    console.log('Latitude:', latCol);
    console.log('Longitude:', lonCol);
    console.log('Tanggal:', dateCol);
    console.log('Tingkat:', levelCol);

    if (!latCol || !lonCol) {
        // Tampilkan semua kolom yang tersedia untuk debugging
        const availableColumns = columns.join(', ');
        throw new Error(`Kolom koordinat tidak ditemukan. Kolom yang tersedia: ${availableColumns}. Pastikan file memiliki kolom koordinat dengan nama seperti 'Koordinat GPS - Lintang' dan 'Koordinat GPS - Bujur'.`);
    }

    // Validate coordinate ranges
    let validCount = 0;
    let invalidCount = 0;

    const processedData = rawData.map((row, index) => {
        const lat = parseFloat(row[latCol]);
        const lon = parseFloat(row[lonCol]);
        
        if (isNaN(lat) || isNaN(lon)) {
            console.warn(`Baris ${index + 1} memiliki koordinat tidak valid:`, row[latCol], row[lonCol]);
            invalidCount++;
            return null;
        }
        
        // Basic coordinate validation (Indonesia bounds: approximately -11 to 6 lat, 95 to 141 lon)
        if (lat < -11 || lat > 6 || lon < 95 || lon > 141) {
            console.warn(`Baris ${index + 1} koordinat di luar wilayah Indonesia:`, lat, lon);
        }
        
        validCount++;
        return {
            latitude: lat,
            longitude: lon,
            tanggal: row[dateCol] || 'N/A',
            tingkat: normalizeSeverity(row[levelCol]),
            id: row.ID || row.id || `row-${index+1}`
        };
    }).filter(d => d !== null && d.latitude && d.longitude);

    if (processedData.length === 0) {
        throw new Error('Tidak ada data dengan koordinat yang valid ditemukan.');
    }

    console.log(`Data diproses: ${validCount} valid, ${invalidCount} tidak valid`);
    return processedData;
}

// Normalisasi tingkat keparahan
function normalizeSeverity(severity) {
    if (!severity) return 'Ringan';
    const s = String(severity).toLowerCase();
    if (s.includes('berat') || s.includes('fatal') || s.includes('meninggal')) return 'Berat';
    if (s.includes('sedang') || s.includes('luka') || s.includes('moderate')) return 'Sedang';
    return 'Ringan';
}

// Menampilkan pratinjau data
function showPreview(file, data) {
    if (!previewSection) return;
    previewSection.style.display = 'block';
    previewSection.classList.add('fade-in');

    const fileInfo = document.getElementById('fileInfo');
    if(fileInfo) {
        const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
        fileInfo.innerHTML = `
            <h4>📊 Informasi File</h4>
            <p><strong>Nama:</strong> ${file.name}</p>
            <p><strong>Ukuran:</strong> ${fileSizeMB} MB</p>
            <p><strong>Total Data:</strong> ${data.length} kejadian kecelakaan</p>
        `;
    }
    
    // Tampilkan statistik tingkat keparahan
    const severityStats = {};
    data.forEach(row => {
        const severity = row.tingkat || 'Tidak Diketahui';
        severityStats[severity] = (severityStats[severity] || 0) + 1;
    });
    
    // Tampilkan validasi dan tabel contoh
    const validationStatus = document.getElementById('validationStatus');
    if (validationStatus) {
        let statsHTML = '<h4>✅ Status Validasi</h4><p>Data terlihat valid dan siap diproses.</p>';
        statsHTML += '<h5>Distribusi Tingkat Keparahan:</h5><ul>';
        Object.entries(severityStats).forEach(([severity, count]) => {
            const percentage = ((count / data.length) * 100).toFixed(1);
            const icon = severity === 'Berat' ? '🔴' : severity === 'Sedang' ? '🟡' : '🟢';
            statsHTML += `<li>${icon} ${severity}: ${count} (${percentage}%)</li>`;
        });
        statsHTML += '</ul>';
        validationStatus.innerHTML = statsHTML;
    }

    const dataSample = document.getElementById('dataSample');
    if (dataSample) {
        const sample = data.slice(0, 5);
        let tableHTML = `<h4>📋 Contoh Data (5 Baris Pertama)</h4>
            <div style="overflow-x: auto;">
            <table class="sample-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Latitude</th>
                        <th>Longitude</th>
                        <th>Tingkat</th>
                        <th>Tanggal</th>
                    </tr>
                </thead>
                <tbody>`;
        
        sample.forEach(row => {
            const severityIcon = row.tingkat === 'Berat' ? '🔴' : row.tingkat === 'Sedang' ? '🟡' : '🟢';
            tableHTML += `
                <tr>
                    <td>${row.id}</td>
                    <td>${row.latitude.toFixed(5)}</td>
                    <td>${row.longitude.toFixed(5)}</td>
                    <td>${severityIcon} ${row.tingkat}</td>
                    <td>${row.tanggal}</td>
                </tr>`;
        });
        tableHTML += `</tbody></table></div>`;
        dataSample.innerHTML = tableHTML;
    }

    if (processSection) {
        processSection.style.display = 'block';
        processSection.classList.add('slide-up');
    }
}

// Mengatur ulang halaman upload
function resetUpload() {
    uploadedData = null;
    currentFile = null;
    isProcessing = false;
    
    if (fileInput) fileInput.value = '';
    if (previewSection) previewSection.style.display = 'none';
    if (processSection) processSection.style.display = 'none';
    if (progressArea) progressArea.style.display = 'none';
    if (resultsArea) resultsArea.style.display = 'none';
    
    // Remove error details if present
    const errorContainer = document.getElementById('errorDetails');
    if (errorContainer) {
        errorContainer.remove();
    }
    
    // Reset tombol
    const processBtn = document.getElementById('processBtn');
    if (processBtn) {
        processBtn.disabled = false;
        processBtn.textContent = 'Mulai Analisis Hotspot';
        processBtn.style.display = 'block';
    }
    
    showNotification('🔄 Halaman telah direset.', 'info');
}

// Menghasilkan file contoh
function generateSampleFile() {
    const sampleData = [
        ['ID', 'Koordinat GPS - Lintang', 'Koordinat GPS - Bujur', 'Tanggal Kejadian', 'Tingkat Kecelakaan'],
        ['KC001', -5.168, 119.465, '2024-01-15', 'Berat'],
        ['KC002', -5.170, 119.470, '2024-02-20', 'Sedang'],
        ['KC003', -5.165, 119.462, '2024-03-10', 'Ringan'],
        ['KC004', -5.172, 119.468, '2024-04-05', 'Berat'],
        ['KC005', -5.169, 119.466, '2024-05-12', 'Sedang'],
        ['KC006', -5.171, 119.464, '2024-06-08', 'Ringan'],
        ['KC007', -5.167, 119.467, '2024-07-22', 'Berat'],
        ['KC008', -5.173, 119.461, '2024-08-14', 'Sedang']
    ];
    
    // Cek apakah XLSX library tersedia
    if (typeof XLSX !== 'undefined') {
        try {
            const wb = XLSX.utils.book_new();
            const ws = XLSX.utils.aoa_to_sheet(sampleData);
            XLSX.utils.book_append_sheet(wb, ws, 'Sample Data');
            XLSX.writeFile(wb, 'sample_data_kecelakaan.xlsx');
            showNotification('📁 File contoh Excel berhasil diunduh!', 'success');
        } catch (error) {
            console.warn('XLSX error, falling back to CSV:', error);
            generateCSVSample(sampleData);
        }
    } else {
        console.warn('XLSX library not available, generating CSV instead');
        generateCSVSample(sampleData);
    }
}

// Helper function untuk generate CSV sample
function generateCSVSample(sampleData) {
    try {
        const csvContent = sampleData.map(row => 
            row.map(cell => `"${cell}"`).join(',')
        ).join('\n');
        
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', 'sample_data_kecelakaan.csv');
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        showNotification('📁 File contoh CSV berhasil diunduh!', 'success');
    } catch (error) {
        console.error('Error generating CSV sample:', error);
        showNotification('❌ Gagal membuat file contoh. Coba lagi.', 'error');
    }
}

// Menampilkan notifikasi dengan styling yang lebih baik
function showNotification(message, type = 'info') {
    // Remove existing notifications to prevent overflow
    const existingNotifications = document.querySelectorAll('.notification');
    existingNotifications.forEach(notification => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    });

    const container = document.body;
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = message; // Use innerHTML to support emojis
    
    const colors = {
        success: '#27ae60',
        error: '#e74c3c',
        warning: '#f39c12',
        info: '#3498db'
    };
    
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${colors[type] || colors.info};
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        z-index: 10000;
        opacity: 0;
        transform: translateX(100%);
        transition: all 0.3s ease;
        max-width: 350px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        font-weight: 500;
        line-height: 1.4;
        word-wrap: break-word;
    `;
    container.appendChild(notification);
    
    setTimeout(() => {
        notification.style.opacity = '1';
        notification.style.transform = 'translateX(0)';
        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (container.contains(notification)) {
                    container.removeChild(notification);
                }
            }, 300);
        }, type === 'error' ? 8000 : 5000); // Error messages stay longer
    }, 10);
}

// Helper untuk update progress bar
function updateProgressBar(phase, task, percentage) {
    const progressPhase = document.getElementById('progressPhase');
    const currentTask = document.getElementById('currentTask');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    
    if (progressPhase) progressPhase.textContent = phase;
    if (currentTask) currentTask.textContent = task;
    if (progressFill) progressFill.style.width = `${Math.min(percentage, 100)}%`;
    if (progressText) progressText.textContent = `${Math.min(Math.round(percentage), 100)}%`;
}

// Menampilkan ringkasan hasil setelah unduhan
function showResultsSummaryAfterDownload(filename) {
    if (!resultsArea) return;
    resultsArea.style.display = 'block';
    resultsArea.classList.add('fade-in');
    
    const resultsStats = document.getElementById('resultsStats');
    if (resultsStats) {
        resultsStats.innerHTML = `
            <div class="results-summary">
                <h3Analisis Hotspot Selesai!</h3>
                <div class="result-grid">
                    <div class="stat-item">
                        <div class="stat-icon"></div>
                        <div class="stat-content">
                            <div class="stat-label">Status</div>
                            <div class="stat-value">Analisis Berhasil</div>
                        </div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-icon"></div>
                        <div class="stat-content">
                            <div class="stat-label">File Hasil</div>
                            <div class="stat-value" style="font-size: 0.9em;">${filename}</div>
                        </div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-icon"></div>
                        <div class="stat-content">
                            <div class="stat-label">WebGIS</div>
                            <div class="stat-value">Siap Digunakan</div>
                        </div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-icon"></div>
                        <div class="stat-content">
                            <div class="stat-label">Data</div>
                            <div class="stat-value">${uploadedData?.length || 0} Kejadian</div>
                        </div>
                    </div>
                </div>
                <div class="next-steps">
                    <h4>Langkah Selanjutnya:</h4>
                    <ol>
                        <li>Buka file ZIP yang telah diunduh</li>
                        <li>Buka file <code>visualisasi_hasil_osmx.html</code> di browser</li>
                        <li>Lihat hasil analisis dalam <code>hasil_analisis_osmx.geojson</code></li>
                        <li>Gunakan data untuk pengambilan keputusan</li>
                    </ol>
                    <div style="margin-top: 15px; padding: 10px; background: #e8f5e8; border-radius: 5px; border-left: 4px solid #27ae60;">
                        <strong>💡 Tips:</strong> File HTML dapat dibuka langsung di browser untuk melihat peta interaktif hasil analisis hotspot.
                    </div>
                </div>
            </div>
        `;
    }
    
    // Sembunyikan tombol proses
    const processBtn = document.getElementById('processBtn');
    if (processBtn) processBtn.style.display = 'none';
}

// Additional utility functions

/**
 * Validate file before processing
 */
function validateFile(file) {
    const validExtensions = ['xlsx', 'csv', 'geojson'];
    const maxSize = 50 * 1024 * 1024; // 50MB
    
    const extension = file.name.toLowerCase().split('.').pop();
    
    if (!validExtensions.includes(extension)) {
        throw new Error(`Format file tidak didukung. Gunakan: ${validExtensions.join(', ')}`);
    }
    
    if (file.size > maxSize) {
        throw new Error(`File terlalu besar (${(file.size/1024/1024).toFixed(1)}MB). Maksimal 50MB.`);
    }
    
    if (file.size === 0) {
        throw new Error('File kosong atau rusak.');
    }
    
    return true;
}

/**
 * Format file size for display
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Check if required libraries are loaded
 */
function checkLibraryDependencies() {
    const dependencies = {
        'XLSX': 'SheetJS library untuk membaca file Excel',
        'Papa': 'PapaParse library untuk membaca file CSV (opsional)'
    };
    
    const missing = [];
    Object.keys(dependencies).forEach(lib => {
        if (typeof window[lib] === 'undefined') {
            missing.push(`${lib}: ${dependencies[lib]}`);
        }
    });
    
    if (missing.length > 0) {
        console.warn('Library dependencies tidak lengkap:', missing);
        // Don't show notification for Papa since it's optional
        if (missing.some(m => m.includes('XLSX'))) {
            showNotification('⚠️ Beberapa fitur mungkin tidak tersedia karena library tidak lengkap.', 'warning');
        }
    }
    
    return missing.length === 0;
}

// Initialize library check when page loads
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(checkLibraryDependencies, 1000);
});

// Export functions for potential use in other scripts
window.HotspotAnalyzer = {
    startProcessing,
    resetUpload,
    generateSampleFile,
    showNotification,
    validateFile,
    formatFileSize
};