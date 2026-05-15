/**
 * NeuroScan AI — Main Frontend Script
 * Explainable AI-Based Alzheimer's Disease Detection System
 * Handles: file upload, drag-drop, AJAX predict, tab switching, sidebar, toast
 */

// ── File Upload State ────────────────────────────────────────────────────────
let selectedFile = null;

// ── DOM Ready ────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initDropZone();
  initSidebarToggle();
});

// ── Sidebar Toggle ───────────────────────────────────────────────────────────
function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  if (sidebar) sidebar.classList.toggle('open');
}

function initSidebarToggle() {
  const toggleBtn = document.getElementById('sidebarToggle');
  if (toggleBtn) toggleBtn.style.display = 'block';
  // Hide sidebar on mobile by default handled by CSS
}

// ── Drop Zone Initialization ─────────────────────────────────────────────────
function initDropZone() {
  const zone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  if (!zone || !fileInput) return;

  // Click anywhere on zone to open file picker (already handled by input overlay)
  // Drag events
  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
  });

  // File input change
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
  });
}

// ── File Handler ─────────────────────────────────────────────────────────────
function handleFile(file) {
  const allowed = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff'];
  if (!allowed.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|bmp|tiff)$/i)) {
    showToast('Invalid file type. Please upload JPG, PNG, BMP, or TIFF.', 'error');
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    showToast('File too large. Maximum size is 16 MB.', 'error');
    return;
  }

  selectedFile = file;

  // Show preview
  const reader = new FileReader();
  reader.onload = e => {
    const previewImg = document.getElementById('previewImg');
    const uploadText = document.getElementById('uploadText');
    if (previewImg) {
      previewImg.src = e.target.result;
      previewImg.classList.add('visible');
    }
    if (uploadText) uploadText.style.display = 'none';
  };
  reader.readAsDataURL(file);

  // Show filename bar
  const fileNameBar = document.getElementById('fileNameBar');
  const fileNameText = document.getElementById('fileNameText');
  if (fileNameBar) fileNameBar.style.display = 'block';
  if (fileNameText) fileNameText.textContent = file.name;

  // Enable button
  const analyzeBtn = document.getElementById('analyzeBtn');
  if (analyzeBtn) analyzeBtn.disabled = false;

  // Style drop zone
  const zone = document.getElementById('dropZone');
  if (zone) zone.classList.add('has-file');

  showToast(`File selected: ${file.name}`, 'success');
}

// ── Clear File ───────────────────────────────────────────────────────────────
function clearFile() {
  selectedFile = null;
  const previewImg = document.getElementById('previewImg');
  const uploadText = document.getElementById('uploadText');
  const fileNameBar = document.getElementById('fileNameBar');
  const analyzeBtn  = document.getElementById('analyzeBtn');
  const fileInput   = document.getElementById('fileInput');
  const zone        = document.getElementById('dropZone');

  if (previewImg)  { previewImg.src = ''; previewImg.classList.remove('visible'); }
  if (uploadText)  uploadText.style.display = '';
  if (fileNameBar) fileNameBar.style.display = 'none';
  if (analyzeBtn)  analyzeBtn.disabled = true;
  if (fileInput)   fileInput.value = '';
  if (zone)        zone.classList.remove('has-file');
}

// ── Analyze Image (AJAX) ─────────────────────────────────────────────────────
function analyzeImage() {
  if (!selectedFile) {
    showToast('Please select an MRI image first.', 'error');
    return;
  }

  const analyzeBtn   = document.getElementById('analyzeBtn');
  const loadingState = document.getElementById('loadingState');

  // Show loading
  if (analyzeBtn)   analyzeBtn.style.display = 'none';
  if (loadingState) loadingState.style.display = 'flex';

  const formData = new FormData();
  formData.append('mri_image', selectedFile);

  fetch('/predict', { method: 'POST', body: formData })
    .then(res => res.json())
    .then(data => {
      if (data.error) {
        showToast(`Error: ${data.error}`, 'error');
        if (analyzeBtn)   analyzeBtn.style.display = '';
        if (loadingState) loadingState.style.display = 'none';
        return;
      }
      // Redirect to result page
      showToast('Analysis complete! Loading results...', 'success');
      setTimeout(() => { window.location.href = '/result'; }, 600);
    })
    .catch(err => {
      showToast('Network error. Please try again.', 'error');
      if (analyzeBtn)   analyzeBtn.style.display = '';
      if (loadingState) loadingState.style.display = 'none';
      console.error(err);
    });
}

// ── Tab Switching ────────────────────────────────────────────────────────────
function switchTab(btn, panelId) {
  // Deactivate all tabs in the same container
  const tabList = btn.closest('.tab-list');
  if (tabList) {
    tabList.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  }
  btn.classList.add('active');

  // Find the parent container and deactivate all panels inside it
  const parent = tabList ? tabList.parentElement : document.body;
  parent.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

  // Activate target panel
  const target = document.getElementById(panelId);
  if (target) target.classList.add('active');
}

// ── Toast Notification ───────────────────────────────────────────────────────
function showToast(message, type = 'info') {
  const toast   = document.getElementById('toast');
  const toastMsg  = document.getElementById('toastMsg');
  const toastIcon = document.getElementById('toastIcon');
  if (!toast || !toastMsg) return;

  toastMsg.textContent = message;

  const icons = { info: 'fa-circle-info', success: 'fa-circle-check', error: 'fa-circle-xmark' };
  const colors = { info: 'var(--teal)', success: 'var(--green)', error: 'var(--red)' };

  if (toastIcon) {
    toastIcon.className = `fa ${icons[type] || icons.info}`;
    toastIcon.style.color = colors[type] || colors.info;
  }

  toast.className = `show ${type}`;

  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { toast.className = ''; }, 3500);
}

// ── Email Report Handler ────────────────────────────────────────────────────
async function sendEmailReport(e) {
  e.preventDefault();
  const form = e.target;
  const btn = document.getElementById('btnSendEmail');
  
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<i class="fa fa-spinner fa-spin"></i> Sending...';
  }
  
  const formData = new FormData(form);
  
  try {
    const res = await fetch('/send_report', {
      method: 'POST',
      body: formData
    });
    
    const data = await res.json();
    
    if (data.success) {
      showToast('Report sent successfully!', 'success');
      form.reset();
    } else {
      showToast(data.message || 'Failed to send report.', 'error');
    }
  } catch (err) {
    showToast('Network error while sending email.', 'error');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<i class="fa fa-paper-plane"></i> Send to Doctor';
    }
  }
}

// ── Theme Toggle ─────────────────────────────────────────────────────────────
function toggleTheme() {
  const body = document.body;
  const icon = document.querySelector('#themeToggle i');
  
  if (body.classList.contains('light-mode')) {
    body.classList.remove('light-mode');
    localStorage.setItem('theme', 'dark');
    if (icon) { icon.classList.remove('fa-sun'); icon.classList.add('fa-moon'); }
  } else {
    body.classList.add('light-mode');
    localStorage.setItem('theme', 'light');
    if (icon) { icon.classList.remove('fa-moon'); icon.classList.add('fa-sun'); }
  }
}

function initTheme() {
  const theme = localStorage.getItem('theme');
  if (theme === 'light') {
    document.body.classList.add('light-mode');
    const icon = document.querySelector('#themeToggle i');
    if (icon) { icon.classList.remove('fa-moon'); icon.classList.add('fa-sun'); }
  }
}

// Call initTheme on load
document.addEventListener('DOMContentLoaded', initTheme);
