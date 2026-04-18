const state = {
  jobId: null,
  uploadedVideoUrl: null,
  pollingHandle: null,
};

const videoFileInput = document.getElementById("videoFile");
const selectedFile = document.getElementById("selectedFile");
const uploadBtn = document.getElementById("uploadBtn");
const processBtn = document.getElementById("processBtn");
const messageBox = document.getElementById("messageBox");
const uploadedVideo = document.getElementById("uploadedVideo");
const placeholderPreview = document.getElementById("placeholderPreview");
const processedVideo = document.getElementById("processedVideo");
const placeholderProcessed = document.getElementById("placeholderProcessed");
const statusLabel = document.getElementById("statusLabel");
const progressText = document.getElementById("progressText");
const progressFill = document.getElementById("progressFill");
const resultsTableBody = document.getElementById("resultsTableBody");

const totalCount = document.getElementById("totalCount");
const carsCount = document.getElementById("carsCount");
const busesCount = document.getElementById("busesCount");
const trucksCount = document.getElementById("trucksCount");
const motorcyclesCount = document.getElementById("motorcyclesCount");
const durationValue = document.getElementById("durationValue");
const downloadCsvBtn = document.getElementById("downloadCsvBtn");

videoFileInput.addEventListener("change", () => {
  const file = videoFileInput.files[0];
  if (!file) return;

  selectedFile.classList.remove("hidden");
  selectedFile.textContent = `Selected file: ${file.name}`;
  showMessage("", false);
});

uploadBtn.addEventListener("click", async () => {
  const file = videoFileInput.files[0];
  if (!file) {
    showMessage("Please select a video file first.", true);
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  uploadBtn.disabled = true;
  processBtn.disabled = true;
  showMessage("Uploading video...", false);

  try {
    const response = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Upload failed.");
    }

    state.jobId = data.job_id;
    state.uploadedVideoUrl = data.uploaded_video_url;

    uploadedVideo.src = `${state.uploadedVideoUrl}?t=${Date.now()}`;
    uploadedVideo.load();
    uploadedVideo.classList.remove("hidden");
    placeholderPreview.classList.add("hidden");

    processBtn.disabled = false;
    showMessage(`Upload successful. Job ID: ${state.jobId}`, false);

    document.getElementById("logUpload").textContent = file.name;
    document.getElementById("logJobId").textContent = state.jobId;
  } catch (error) {
    showMessage(error.message || "Upload failed.", true);
  } finally {
    uploadBtn.disabled = false;
  }
});

processBtn.addEventListener("click", async () => {
  if (!state.jobId) {
    showMessage("Please upload a video first.", true);
    return;
  }

  processBtn.disabled = true;
  showMessage("Starting processing...", false);

  try {
    const response = await fetch(`/api/process/${state.jobId}`, {
      method: "POST",
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Failed to start processing.");
    }

    showMessage(data.message || "Processing started.", false);
    startPolling();
  } catch (error) {
    showMessage(error.message || "Failed to start processing.", true);
    processBtn.disabled = false;
  }
});

function startPolling() {
  if (state.pollingHandle) clearInterval(state.pollingHandle);

  state.pollingHandle = setInterval(async () => {
    try {
      const response = await fetch(`/api/status/${state.jobId}`);
      const data = await response.json();

      if (response.status === 404) {
        clearInterval(state.pollingHandle);
        showMessage("The server restarted and the job was lost. Please upload again.", true);
        processBtn.disabled = false;
        state.jobId = null;
        return;
      }

      if (!response.ok) {
        throw new Error(data.detail || "Failed to fetch status.");
      }

      updateProgress(data.status, data.progress);

      if (data.status === "completed") {
        clearInterval(state.pollingHandle);
        await loadResults();
      } else if (data.status === "failed") {
        clearInterval(state.pollingHandle);
        showMessage(data.error || "Processing failed.", true);
        processBtn.disabled = false;
      }
    } catch (error) {
      clearInterval(state.pollingHandle);
      showMessage(error.message || "Polling failed.", true);
      processBtn.disabled = false;
    }
  }, 2000);
}

async function loadResults() {
  const response = await fetch(`/api/results/${state.jobId}`);
  const data = await response.json();

  if (!response.ok) {
    showMessage(data.detail || "Failed to load results.", true);
    return;
  }

  const result = data.result;
  const breakdown = result.breakdown_by_type || {};

  totalCount.textContent = result.total_vehicle_count ?? 0;
  carsCount.textContent = breakdown.car ?? 0;
  busesCount.textContent = breakdown.bus ?? 0;
  trucksCount.textContent = breakdown.truck ?? 0;
  motorcyclesCount.textContent = breakdown.motorcycle ?? 0;
  durationValue.textContent = `${result.processing_duration_seconds ?? 0}s`;

  processedVideo.pause();
  processedVideo.removeAttribute("src");
  processedVideo.load();

  processedVideo.src = `${result.processed_video_url}?t=${Date.now()}`;
  processedVideo.classList.remove("hidden");
  placeholderProcessed.classList.add("hidden");
  processedVideo.load();

  downloadCsvBtn.href = `/api/download/report/${state.jobId}?format=csv`;
  downloadCsvBtn.classList.remove("disabled-link");

  renderTable(result.detections || []);
  showMessage("Processing completed successfully.", false);
}

function renderTable(rows) {
  if (!rows.length) {
    resultsTableBody.innerHTML =
      `<tr><td colspan="9" class="empty-cell">No counted vehicle events found.</td></tr>`;
    return;
  }

  resultsTableBody.innerHTML = rows.map((row) => `
    <tr>
      <td>${safe(row.vehicle_id)}</td>
      <td>${safe(row.vehicle_type)}</td>
      <td>${safe(row.crossed_line)}</td>
      <td>${safe(row.first_detected_frame)}</td>
      <td>${safe(row.first_detected_timestamp_seconds)}</td>
      <td>${safe(row.counted_frame)}</td>
      <td>${safe(row.counted_timestamp_seconds)}</td>
      <td>${safe(row.last_detected_frame)}</td>
      <td>${safe(row.last_detected_timestamp_seconds)}</td>
    </tr>
  `).join("");
}

function updateProgress(status, progress) {
  statusLabel.textContent = capitalize(status || "idle");
  progressText.textContent = `${progress || 0}%`;
  progressFill.style.width = `${progress || 0}%`;
}

function showMessage(message, isError) {
  messageBox.textContent = message;
  messageBox.classList.remove("success", "error");
  if (message) {
    messageBox.classList.add(isError ? "error" : "success");
  }
}

function capitalize(value) {
  if (!value) return "";
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function safe(value) {
  return value === undefined || value === null ? "" : value;
}