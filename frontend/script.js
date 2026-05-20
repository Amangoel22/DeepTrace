// ========================================
// DeepTrace — FINAL FIXED VERSION
// ========================================

document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

function $(id) {
    return document.getElementById(id);
}

function initApp() {
    const themeToggle = $('themeToggle');
    const API_BASE_URL = 'http://localhost:8000';

    const uploadZone = $('uploadZone');
    const fileInput = $('fileInput');
    const previewArea = $('previewArea');
    const previewContent = $('previewContent');
    const analyzeBtn = $('analyzeBtn');
    const loadingState = $('loadingState');
    const resultsSection = $('results');

    let currentFile = null;

    // Dark/Light Theme
    function initTheme() {
        const savedTheme = localStorage.getItem('theme') || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);
    }

    function toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    }
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
    initTheme();

    // ================= FILE =================
    if (uploadZone && fileInput) {
        uploadZone.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
    }

    function handleFile(file) {
        currentFile = file;
        showPreview(file);
    }

    function showPreview(file) {
        uploadZone.style.display = 'none';
        previewArea.style.display = 'block';
        previewContent.innerHTML = '';

        if (file.type.startsWith('video/')) {
            const video = document.createElement('video');
            video.src = URL.createObjectURL(file);
            video.controls = true;
            previewContent.appendChild(video);
        } else {
            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            previewContent.appendChild(img);
        }
    }

    // ================= ANALYSIS =================
    if (analyzeBtn) analyzeBtn.addEventListener('click', startAnalysis);

    async function startAnalysis() {
        if (!currentFile) return alert("Upload file");

        previewArea.style.display = 'none';
        loadingState.style.display = 'block';
        resultsSection.style.display = 'none';

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch(`${API_BASE_URL}/analyze`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            displayResults(result);

        } catch (err) {
            console.error(err);
            loadingState.style.display = 'none';
            previewArea.style.display = 'block';
            alert("Server error. Try smaller video.");
        }
    }

    // ================= RESULTS =================
    function displayResults(result) {

        console.log("RESULT:", result); // DEBUG

        if (loadingState) loadingState.style.display = 'none';
        if (resultsSection) resultsSection.style.display = 'block';

        try {

            // Verdict
            if ($('verdictText')) {
                $('verdictText').textContent =
                    result?.model_prediction?.verdict ?? 'UNKNOWN';
            }

            // Confidence
            if ($('confidenceValue')) {
                $('confidenceValue').textContent =
                    result?.authenticity?.score ?? 0;
            }

            // Metadata
            if ($('metaFilename')) $('metaFilename').textContent = result?.filename || '';
            if ($('metaFrames')) $('metaFrames').textContent = result?.total_frames_analyzed || '';
            if ($('metaDuration')) $('metaDuration').textContent = result?.duration || '';
            if ($('metaTimestamp')) $('metaTimestamp').textContent = new Date().toLocaleString();

            // Heatmap
            const heatmapCard = $('heatmapCard');
            const heatmapImg = $('heatmapImage');

            if (result?.heatmap && heatmapCard && heatmapImg) {
                heatmapCard.style.display = 'block';
                heatmapImg.src = 'data:image/jpeg;base64,' + result.heatmap;
            } else if (heatmapCard) {
                heatmapCard.style.display = 'none';
            }

            // Reasons
            const insightsList = $('insightsList');
            if (insightsList) {
                insightsList.innerHTML = '';

                if (result?.reasons?.length > 0) {
                    result.reasons.forEach(r => {
                        const li = document.createElement('li');
                        li.textContent = r;
                        insightsList.appendChild(li);
                    });
                }
            }

            // Segments
            const segmentsCard = $('segmentsCard');
            const segmentsList = $('segmentsList');

            if (result?.fake_segments?.length > 0) {
                segmentsCard.style.display = 'block';
                segmentsList.innerHTML = '';

                result.fake_segments.forEach(seg => {
                    const div = document.createElement('div');
                    div.textContent = `${seg.start}s - ${seg.end}s`;
                    segmentsList.appendChild(div);
                });
            } else if (segmentsCard) {
                segmentsCard.style.display = 'none';
            }

            // Timeline
            const timelineCard = $('timelineCard');
            if (result?.confidence_over_time?.length > 0) {
                timelineCard.style.display = 'block';
                renderTimeline(result.confidence_over_time);
            } else if (timelineCard) {
                timelineCard.style.display = 'none';
            }

            // Breakdown (SAFE)
            if (result?.breakdown) {
                updateBar("blink", result.breakdown.blink || 0);
                updateBar("lip", result.breakdown.lip_sync || 0);
                updateBar("drift", result.breakdown.feature_drift || 0);
            }

        } catch (err) {
            console.error("DISPLAY ERROR:", err);
            alert("Error displaying results. Check console.");
        }
    }

    function renderTimeline(data) {
        const canvas = document.getElementById('confidenceCanvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');

        canvas.width = 500;
        canvas.height = 300;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.beginPath();
        ctx.strokeStyle = '#6366F1';
        ctx.lineWidth = 2;

        data.forEach((point, i) => {
            const x = (i / data.length) * canvas.width;
            const y = canvas.height - (point.probability * canvas.height);

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });

        ctx.stroke();
    }
    function updateBar(id, value) {
        const bar = document.getElementById(id + "Bar");
        const label = document.getElementById(id + "Value");

        if (!bar || !label) return;

        const percent = Math.round((value || 0) * 100);

        bar.style.width = percent + "%";
        label.textContent = percent + "%";
    }
    // ================= REPORT =================
    const reportBtn = $('downloadReport');

    if (reportBtn) {
        reportBtn.addEventListener('click', async () => {

            if (!currentFile) return alert("No file");

            const formData = new FormData();
            formData.append('file', currentFile);

            const response = await fetch(`${API_BASE_URL}/report`, {
                method: 'POST',
                body: formData
            });

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = url;
            a.download = 'DeepTrace_Report.pdf';
            a.click();
        });
    }

    // ================= RESET =================
    const newAnalysisBtn = $('newAnalysis');

    if (newAnalysisBtn) {
        newAnalysisBtn.addEventListener('click', () => {
            location.reload();
        });
    }
}