// Wait for the DOM content to be fully loaded
document.addEventListener("DOMContentLoaded", function () {
  const startButton = document.getElementById("startCapture");
  const stopButton = document.getElementById("stopCapture");

  const useServerCheckbox = document.getElementById("useServerCheckbox");
  const useMultilingualCheckbox = document.getElementById('useMultilingualCheckbox');
  const languageDropdown = document.getElementById('languageDropdown');
  const taskDropdown = document.getElementById('taskDropdown');
  let selectedLanguage = null;
  let selectedTask = taskDropdown.value;

  // Add click event listeners to the buttons
  startButton.addEventListener("click", startCapture);
  stopButton.addEventListener("click", stopCapture);

  // Retrieve capturing state from storage on popup open
  chrome.storage.local.get("capturingState", ({ capturingState }) => {
    if (capturingState && capturingState.isCapturing) {
      toggleCaptureButtons(true);
    } else {
      toggleCaptureButtons(false);
    }
  });

  // Retrieve checkbox state from storage on popup open
  chrome.storage.local.get("useServerState", ({ useServerState }) => {
    if (useServerState !== undefined) {
      useServerCheckbox.checked = useServerState;
    }
  });

  chrome.storage.local.get("useMultilingualModelState", ({ useMultilingualModelState }) => {
    if (useMultilingualModelState !== undefined) {
      useMultilingualCheckbox.checked = useMultilingualModelState;
      languageDropdown.disabled = !useMultilingualModelState;
      taskDropdown.disabled = !useMultilingualModelState;
    }
  });

  chrome.storage.local.get("selectedLanguage", ({ selectedLanguage: storedLanguage }) => {
    if (storedLanguage !== undefined) {
      languageDropdown.value = storedLanguage;
      selectedLanguage = storedLanguage;
    }
  });

  chrome.storage.local.get("selectedTask", ({ selectedTask: storedTask }) => {
    if (storedTask !== undefined) {
      taskDropdown.value = storedTask;
      selectedTask = storedTask;
    }
  });

  // Function to handle the start capture button click event
  async function startCapture() {
    // Ignore click if the button is disabled
    if (startButton.disabled) {
      return;
    }

    // Get the current active tab
    const currentTab = await getCurrentTab();

    // Send a message to the background script to start capturing
    let host = "localhost";
    let port = "9090";
    const useCollaboraServer = useServerCheckbox.checked;
    if (useCollaboraServer){
      host = "transcription.kurg.org"
      port = "7090"
    }

    chrome.runtime.sendMessage(
      { 
        action: "startCapture", 
        tabId: currentTab.id,
        host: host,
        port: port,
        useMultilingual: useMultilingualCheckbox.checked,
        language: selectedLanguage,
        task: selectedTask
      }, () => {
        // Update capturing state in storage and toggle the buttons
        chrome.storage.local.set({ capturingState: { isCapturing: true } }, () => {
          toggleCaptureButtons(true);
        });
      }
    );
  }

  // Function to handle the stop capture button click event
  function stopCapture() {
    // Ignore click if the button is disabled
    if (stopButton.disabled) {
      return;
    }

    // Send a message to the background script to stop capturing
    chrome.runtime.sendMessage({ action: "stopCapture" }, () => {
      // Update capturing state in storage and toggle the buttons
      chrome.storage.local.set({ capturingState: { isCapturing: false } }, () => {
        toggleCaptureButtons(false);
      });
    });
  }

  // Function to get the current active tab
  async function getCurrentTab() {
    return new Promise((resolve) => {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        resolve(tabs[0]);
      });
    });
  }

  // Function to toggle the capture buttons based on the capturing state
  function toggleCaptureButtons(isCapturing) {
    startButton.disabled = isCapturing;
    stopButton.disabled = !isCapturing;
    useServerCheckbox.disabled = isCapturing; 
    startButton.classList.toggle("disabled", isCapturing);
    stopButton.classList.toggle("disabled", !isCapturing);
  }

  // Save the checkbox state when it's toggled
  useServerCheckbox.addEventListener("change", () => {
    const useServerState = useServerCheckbox.checked;
    chrome.storage.local.set({ useServerState });
  });

  useMultilingualCheckbox.addEventListener('change', function() {
    const useMultilingualModelState = useMultilingualCheckbox.checked;
    if (useMultilingualModelState) {
      languageDropdown.disabled = false;
      taskDropdown.disabled = false;
    } else {
      languageDropdown.disabled = true;
      taskDropdown.disabled = true;
    }
    chrome.storage.local.set({ useMultilingualModelState });
  });

  languageDropdown.addEventListener('change', function() {
    if (languageDropdown.value === "") {
      selectedLanguage = null;
    } else {
      selectedLanguage = languageDropdown.value;
    }
    chrome.storage.local.set({ selectedLanguage });
  });

  taskDropdown.addEventListener('change', function() {
    selectedTask = taskDropdown.value;
    chrome.storage.local.set({ selectedTask });
  });

  chrome.runtime.onMessage.addListener(async (request, sender, sendResponse) => {
    if (request.action === "updateSelectedLanguage") {
      const detectedLanguage = request.detectedLanguage;
  
      if (detectedLanguage) {
        languageDropdown.value = detectedLanguage;
        chrome.storage.local.set({ selectedLanguage: detectedLanguage });
      }
    }
  });
  
});
