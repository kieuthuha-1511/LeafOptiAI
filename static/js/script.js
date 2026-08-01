document.addEventListener('DOMContentLoaded', () => {
    // ==========================================
    // 0. TỰ ĐỘNG CHÈN CSS TỐI ƯU GIAO DIỆN
    // ==========================================
    const injectGlobalStyles = () => {
        if (document.getElementById('custom-injected-styles')) return;
        const style = document.createElement('style');
        style.id = 'custom-injected-styles';
        style.innerHTML = `
            #origImg:not([src]), #origImg[src=""],
            #segImg:not([src]), #segImg[src=""],
            #maskImg:not([src]), #maskImg[src=""] {
                display: none !important;
            }
            #origImg, #segImg, #maskImg {
                width: 100% !important;
                height: 260px !important;
                object-fit: contain !important;
                border-radius: 8px;
            }
            .viewer-button.viewer-close {
                background-color: #ff4d4d !important;
                opacity: 0.95 !important;
            }
            .viewer-button.viewer-close:hover {
                background-color: #ff0000 !important;
                opacity: 1 !important;
                transform: scale(1.1);
                transition: all 0.2s ease;
            }
            .history-card {
                background: #ffffff;
                border-radius: 12px;
                border: 1px solid #e0e0e0;
                box-shadow: 0 4px 12px rgba(0,0,0,0.03);
            }
            .history-table th {
                background-color: #1e5631 !important;
                color: #ffffff !important;
                font-weight: 600;
                font-size: 0.88rem;
                vertical-align: middle;
            }
            .history-table td {
                font-size: 0.85rem;
                vertical-align: middle;
            }
            .history-scroll-wrapper {
                max-height: 280px;
                overflow-y: auto;
            }
        `;
        document.head.appendChild(style);
    };
    injectGlobalStyles();

    // === Các element cơ bản ===
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    
    const origImg = document.getElementById('origImg');
    const segImg = document.getElementById('segImg');
    const maskImg = document.getElementById('maskImg');
    
    const scanOverlay = document.getElementById('scanOverlay');
    const scanLine = document.getElementById('scanLine');
    
    const valLeafArea = document.getElementById('valLeafArea');
    const valCoverage = document.getElementById('valCoverage');
    const barCoverage = document.getElementById('barCoverage');
    const valRegions = document.getElementById('valRegions');
    const valConfidence = document.getElementById('valConfidence');
    const valTime = document.getElementById('valTime');
    
    const btnAnalyze = document.getElementById('btnAnalyze');
    const btnClear = document.getElementById('btnClear');
    const btnExportExcel = document.getElementById('btnExportExcel') || document.getElementById('btnExportCSV');
    const btnExportPDF = document.getElementById('btnExportPDF');
    const btnDownloadPDF = document.getElementById('btnDownloadPDF');

    let selectedFile = null;
    let currentAnalysisData = null;

    // ==========================================
    // LOCAL STORAGE & QUẢN LÝ LỊCH SỬ
    // ==========================================
    const STORAGE_KEY = 'leafopti_analysis_history';

    const loadHistoryFromStorage = () => {
        try {
            const data = localStorage.getItem(STORAGE_KEY);
            return data ? JSON.parse(data) : [];
        } catch (e) {
            console.error('Lỗi đọc LocalStorage:', e);
            return [];
        }
    };

    const saveHistoryToStorage = () => {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(analysisHistory));
        } catch (e) {
            console.error('Lỗi ghi LocalStorage:', e);
        }
    };

    let analysisHistory = loadHistoryFromStorage();

    let origViewer = null;
    let segViewer = null;
    let maskViewer = null;

    const defaultViewerOptions = {
        button: true,
        toolbar: true,
        navbar: false,
        title: false,
        tooltip: false,
        movable: true,
        zoomable: true,
        rotatable: false,
        scalable: false
    };

    const createViewer = (imgElement) => {
        if (typeof Viewer !== 'undefined' && imgElement) {
            return new Viewer(imgElement, defaultViewerOptions);
        }
        return null;
    };

    const escapeHtml = (str) => {
        return String(str || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    };

    const initHistoryUI = () => {
        const btnClearHistory = document.getElementById('btnClearHistory');
        if (btnClearHistory) {
            btnClearHistory.onclick = () => {
                if (analysisHistory.length === 0) return;
                if (confirm('Bạn có chắc chắn muốn xóa toàn bộ lịch sử phân tích?')) {
                    analysisHistory = [];
                    saveHistoryToStorage();
                    renderHistoryTable();
                }
            };
        }
        renderHistoryTable();
    };

    function renderHistoryTable() {
        const tbody = document.getElementById('historyTableBody');
        if (!tbody) return;

        if (analysisHistory.length === 0) {
            tbody.innerHTML = `
                <tr id="emptyHistoryRow">
                    <td colspan="6" class="text-muted py-3">Chưa có lịch sử phân tích nào</td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = analysisHistory.map((item, index) => `
            <tr>
                <td><strong>${index + 1}</strong></td>
                <td class="text-start text-truncate" style="max-width: 220px;" title="${escapeHtml(item.fileName)}">
                    <i class="ri-image-line me-1 text-primary"></i>${escapeHtml(item.fileName)}
                </td>
                <td class="text-muted">${item.timestamp}</td>
                <td><span class="badge bg-light text-dark border fw-bold">${item.leafArea}</span></td>
                <td><span class="badge bg-success-subtle text-success border border-success fw-bold">${item.coverage}%</span></td>
                <td><span class="badge bg-info-subtle text-info border border-info fw-bold">${item.confidence}%</span></td>
            </tr>
        `).join('');
    }

    initHistoryUI();

    // ==========================================
    // 1. XỬ LÝ UPLOAD & KÉO THẢ
    // ==========================================
    if (uploadZone) {
        uploadZone.addEventListener('click', () => fileInput && fileInput.click());
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover', 'drag-active');
        });
        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover', 'drag-active');
        });
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover', 'drag-active');
            if (e.dataTransfer.files.length > 0) {
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });
    }

    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('Vui lòng chọn file hình ảnh hợp lệ.');
            return;
        }

        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            origImg.src = e.target.result;
            origImg.style.display = 'block';
            
            segImg.removeAttribute('src');
            maskImg.removeAttribute('src');
            segImg.style.display = 'none';
            maskImg.style.display = 'none';

            resetStats();
            destroyViewers();
            origViewer = createViewer(origImg);
        };
        reader.readAsDataURL(file);
    }

    // ==========================================
    // 2. NÚT ANALYZE
    // ==========================================
    if (btnAnalyze) {
        btnAnalyze.addEventListener('click', async () => {
            if (!selectedFile) {
                alert('Vui lòng tải lên ảnh nhiệt trước khi bấm Analyze.');
                return;
            }

            const formData = new FormData();
            formData.append('file', selectedFile);

            btnAnalyze.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status"></span> Analyzing...`;
            btnAnalyze.disabled = true;
            
            if (scanOverlay && scanLine) {
                scanOverlay.style.display = 'flex';
                scanLine.style.display = 'block';
            }

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.success) {
                    origImg.src = result.images.original;
                    segImg.src = result.images.segmentation;
                    maskImg.src = result.images.binary_mask;

                    origImg.style.display = 'block';
                    segImg.style.display = 'block';
                    maskImg.style.display = 'block';

                    const m = result.metrics;
                    if (valLeafArea) valLeafArea.textContent = m.leaf_area.toLocaleString();
                    if (valCoverage) valCoverage.textContent = m.coverage;
                    if (barCoverage) barCoverage.style.width = `${m.coverage}%`;
                    if (valRegions) valRegions.textContent = m.detected_regions;
                    if (valConfidence) valConfidence.textContent = (m.confidence).toFixed(1);
                    if (valTime) valTime.textContent = m.inference_time;

                    currentAnalysisData = result;

                    const now = new Date();
                    const timeString = now.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' }) + ' ' + now.toLocaleDateString('vi-VN');
                    
                    const newHistoryEntry = {
                        id: Date.now(),
                        fileName: selectedFile.name,
                        timestamp: timeString,
                        leafArea: m.leaf_area.toLocaleString(),
                        coverage: m.coverage,
                        confidence: (m.confidence).toFixed(1),
                        detectedRegions: m.detected_regions,
                        inferenceTime: m.inference_time
                    };

                    analysisHistory.unshift(newHistoryEntry);
                    saveHistoryToStorage();
                    renderHistoryTable();

                    destroyViewers();
                    origViewer = createViewer(origImg);
                    segViewer = createViewer(segImg);
                    maskViewer = createViewer(maskImg);

                } else {
                    alert('Lỗi phân tích: ' + result.error);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Không thể kết nối tới server. Vui lòng kiểm tra lại!');
            } finally {
                btnAnalyze.innerHTML = `<i class="ri-brain-line me-2"></i> Analyze`;
                btnAnalyze.disabled = false;
                
                if (scanOverlay && scanLine) {
                    scanOverlay.style.display = 'none';
                    scanLine.style.display = 'none';
                }
            }
        });
    }

    // ==========================================
    // 3. NÚT CLEAR
    // ==========================================
    if (btnClear) {
        btnClear.addEventListener('click', () => {
            selectedFile = null;
            currentAnalysisData = null;
            if (fileInput) fileInput.value = '';

            origImg.removeAttribute('src');
            segImg.removeAttribute('src');
            maskImg.removeAttribute('src');

            origImg.style.display = 'none';
            segImg.style.display = 'none';
            maskImg.style.display = 'none';

            if (scanOverlay && scanLine) {
                scanOverlay.style.display = 'none';
                scanLine.style.display = 'none';
            }

            destroyViewers();
            resetStats();
        });
    }

    function resetStats() {
        if (valLeafArea) valLeafArea.textContent = '--';
        if (valCoverage) valCoverage.textContent = '--';
        if (barCoverage) barCoverage.style.width = '0%';
        if (valRegions) valRegions.textContent = '--';
        if (valConfidence) valConfidence.textContent = '--';
        if (valTime) valTime.textContent = '--';
        if (btnDownloadPDF) {
            btnDownloadPDF.classList.add('disabled');
            btnDownloadPDF.onclick = null;
        }
    }

    function destroyViewers() {
        if (origViewer) { origViewer.destroy(); origViewer = null; }
        if (segViewer) { segViewer.destroy(); segViewer = null; }
        if (maskViewer) { maskViewer.destroy(); maskViewer = null; }
    }

    // ==========================================
    // 4. XUẤT FILE EXCEL ĐỊNH DẠNG .XLSX (CHUYÊN NGHIỆP)
    // ==========================================
    if (btnExportExcel) {
        btnExportExcel.addEventListener('click', () => {
            if (analysisHistory.length === 0 && !currentAnalysisData) {
                alert('Chưa có dữ liệu phân tích nào trong lịch sử để xuất Excel.');
                return;
            }

            // Chuẩn bị dữ liệu danh sách dạng mảng cấu trúc cho SheetJS
            const excelData = [];
            excelData.push(["STT", "Tên ảnh", "Thời gian phân tích", "Diện tích lá (px)", "Tỷ lệ phủ (%)", "Độ tin cậy (%)"]);

            if (analysisHistory.length > 0) {
                analysisHistory.forEach((item, index) => {
                    excelData.push([
                        index + 1,
                        item.fileName,
                        item.timestamp,
                        item.leafArea,
                        parseFloat(item.coverage),
                        parseFloat(item.confidence)
                    ]);
                });
            } else if (currentAnalysisData) {
                const m = currentAnalysisData.metrics;
                const fileName = selectedFile ? selectedFile.name : "Ảnh phân tích";
                const now = new Date().toLocaleString('vi-VN');
                excelData.push([
                    1,
                    fileName,
                    now,
                    m.leaf_area,
                    parseFloat(m.coverage),
                    parseFloat((m.confidence * 100).toFixed(1))
                ]);
            }

            // Kiểm tra xem thư viện XLSX (SheetJS) đã được nhúng chưa, nếu có dùng định dạng .xlsx chuẩn
            if (typeof XLSX !== 'undefined') {
                const worksheet = XLSX.utils.aoa_to_sheet(excelData);
                
                // Tự động căn chỉnh độ rộng cột tối ưu
                worksheet['!cols'] = [
                    { wch: 6 },  // STT
                    { wch: 25 }, // Tên ảnh
                    { wch: 22 }, // Thời gian
                    { wch: 18 }, // Diện tích
                    { wch: 15 }, // Tỷ lệ phủ
                    { wch: 15 }  // Độ tin cậy
                ];

                const workbook = XLSX.utils.book_new();
                XLSX.utils.book_append_sheet(workbook, worksheet, "Lịch sử phân tích");
                
                const timestamp = new Date().toISOString().slice(0,10);
                XLSX.writeFile(workbook, `Bao_Cao_Phan_Tich_La_${timestamp}.xlsx`);
            } else {
                // Fallback an toàn nếu chưa nạp thư viện XLSX trên HTML
                let csvContent = "\uFEFF" + excelData.map(e => e.join(",")).join("\n");
                const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                const link = document.createElement("a");
                const url = URL.createObjectURL(blob);
                link.setAttribute("href", url);
                link.setAttribute("download", `Bao_Cao_Phan_Tich_La_${new Date().toISOString().slice(0,10)}.csv`);
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                URL.revokeObjectURL(url);
            }
        });
    }

    // ==========================================
    // 5. XUẤT FILE PDF (KHỚP BACKEND - 1 LẦN GẦN NHẤT)
    // ==========================================
    if (btnExportPDF) {
        btnExportPDF.addEventListener('click', async () => {
            if (!currentAnalysisData && analysisHistory.length === 0) {
                alert('Chưa có dữ liệu phân tích nào để xuất báo cáo PDF.');
                return;
            }

            btnExportPDF.innerHTML = `<span class="spinner-border spinner-border-sm me-1" role="status"></span> Exporting...`;
            btnExportPDF.disabled = true;

            try {
                // Lấy thông tin lần phân tích gần nhất
                const latestItem = analysisHistory.length > 0 ? analysisHistory[0] : null;
                const metricsObj = currentAnalysisData ? currentAnalysisData.metrics : (latestItem ? {
                    leaf_area: parseFloat(String(latestItem.leafArea).replace(/,/g, '')),
                    coverage: parseFloat(latestItem.coverage),
                    detected_regions: latestItem.detectedRegions,
                    confidence: parseFloat(latestItem.confidence) / 100,
                    inference_time: latestItem.inferenceTime
                } : {});

                const payload = {
                    metrics: metricsObj,
                    filenames: [latestItem ? latestItem.fileName : (selectedFile ? selectedFile.name : "image.jpg")],
                    history: latestItem ? [latestItem] : []
                };

                const response = await fetch('/export_pdf', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const result = await response.json();

                if (result.success) {
                    if (btnDownloadPDF) {
                        btnDownloadPDF.classList.remove('disabled');
                        btnDownloadPDF.onclick = () => {
                            window.location.href = result.pdf_url;
                        };
                    }
                    alert('Báo cáo PDF của lần phân tích gần nhất đã sẵn sàng để tải xuống!');
                } else {
                    alert('Không thể xuất PDF: ' + (result.error || 'Lỗi không xác định'));
                }
            } catch (err) {
                console.error(err);
                alert('Có lỗi xảy ra khi tạo PDF. Đảm bảo Backend Server đang hoạt động!');
            } finally {
                btnExportPDF.innerHTML = `<i class="ri-file-pdf-line text-danger me-1"></i> Export PDF`;
                btnExportPDF.disabled = false;
            }
        });
    }
});