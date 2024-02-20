document.addEventListener("DOMContentLoaded", function() {
  const startButton = document.getElementById("startCapture");
  const stopButton = document.getElementById("stopCapture");

  const useServerCheckbox = document.getElementById("useServerCheckbox");
  const useVadCheckbox = document.getElementById("useVadCheckbox");
  const languageDropdown = document.getElementById('languageDropdown');
  const taskDropdown = document.getElementById('taskDropdown');
  const modelSizeDropdown = document.getElementById('modelSizeDropdown');
  let selectedLanguage = null;
  let selectedTask = taskDropdown.value;
  let selectedModelSize = modelSizeDropdown.value;
  

  browser.storage.local.get("capturingState")
    .then(function(result) {
      const capturingState = result.capturingState;
      if (capturingState && capturingState.isCapturing) {
        toggleCaptureButtons(true);
      } else {
        toggleCaptureButtons(false);
      }
      // Enable the startButton
      startButton.disabled = false;
    })
    .catch(function(error) {
      console.error("Error retrieving capturing state:", error);
      // Enable the startButton
      startButton.disabled = false;
    });
  
  browser.storage.local.get("useServerState", ({ useServerState }) => {
    if (useServerState !== undefined) {
      useServerCheckbox.checked = useServerState;
    }
  });

  browser.storage.local.get("useVadState", ({ useVadState }) => {
    if (useVadState !== undefined) {
      useVadCheckbox.checked = useVadState;
    }
  });

  browser.storage.local.get("selectedLanguage", ({ selectedLanguage: storedLanguage }) => {
    if (storedLanguage !== undefined) {
      languageDropdown.value = storedLanguage;
      selectedLanguage = storedLanguage;
    }
  });

  browser.storage.local.get("selectedTask", ({ selectedTask: storedTask }) => {
    if (storedTask !== undefined) {
      taskDropdown.value = storedTask;
      selectedTask = storedTask;
    }
  });

  browser.storage.local.get("selectedModelSize", ({ selectedModelSize: storedModelSize }) => {
    if (storedModelSize !== undefined) {
      modelSizeDropdown.value = storedModelSize;
      selectedModelSize = storedModelSize;
    }
  });

  startButton.addEventListener("click", function() {
    let host = "localhost";
    let port = "9090";
    const useCollaboraServer = useServerCheckbox.checked;

    if (useCollaboraServer){
      host = "transcription.kurg.org"
      port = "7090"
    }

    browser.tabs.query({ active: true, currentWindow: true })
      .then(function(tabs) {
        browser.tabs.sendMessage(
          tabs[0].id, 
          { 
            action: "startCapture", 
            data: {
              host: host,
              port: port,
              language: selectedLanguage,
              task: selectedTask,
              modelSize: selectedModelSize,
              useVad: useVadCheckbox.checked,
            } 
          });
        toggleCaptureButtons(true);
        browser.storage.local.set({ capturingState: { isCapturing: true } })
          .catch(function(error) {
            console.error("Error storing capturing state:", error);
          });
      })
      .catch(function(error) {
        console.error("Error sending startCapture message:", error);
      });
  });

  stopButton.addEventListener("click", function() {
    browser.tabs.query({ active: true, currentWindow: true })
      .then(function(tabs) {
        browser.tabs.sendMessage(tabs[0].id, { action: "stopCapture" })
          .then(function(response) {
            toggleCaptureButtons(false);
            browser.storage.local.set({ capturingState: { isCapturing: false } })
              .catch(function(error) {
                console.error("Error storing capturing state:", error);
              });
          })
          .catch(function(error) {
            console.error("Error sending stopCapture message:", error);
          });
      })
      .catch(function(error) {
        console.error("Error querying active tab:", error);
      });
  });

  // Function to toggle the capture buttons
  function toggleCaptureButtons(isCapturing) {
    startButton.disabled = isCapturing;
    stopButton.disabled = !isCapturing;
    useServerCheckbox.disabled = isCapturing;
    useVadCheckbox.disabled = isCapturing;
    modelSizeDropdown.disabled = isCapturing;
    languageDropdown.disabled = isCapturing;
    taskDropdown.disabled = isCapturing; 
    startButton.classList.toggle("disabled", isCapturing);
    stopButton.classList.toggle("disabled", !isCapturing);
  }

  // Save the checkbox state when it's toggled
  useServerCheckbox.addEventListener("change", () => {
    const useServerState = useServerCheckbox.checked;
    browser.storage.local.set({ useServerState });
  });

  useVadCheckbox.addEventListener("change", () => {
    const useVadState = useVadCheckbox.checked;
    browser.storage.local.set({ useVadState });
  });

  languageDropdown.addEventListener('change', function() {
    if (languageDropdown.value === "") {
      selectedLanguage = null;
    } else {
      selectedLanguage = languageDropdown.value;
    }
    browser.storage.local.set({ selectedLanguage });
  });

  taskDropdown.addEventListener('change', function() {
    selectedTask = taskDropdown.value;
    browser.storage.local.set({ selectedTask });
  });

  modelSizeDropdown.addEventListener('change', function() {
    selectedModelSize = modelSizeDropdown.value;
    browser.storage.local.set({ selectedModelSize });
  });

  browser.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "updateSelectedLanguage") {
      const detectedLanguage = request.data;
  
      if (detectedLanguage) {
        languageDropdown.value = detectedLanguage;
        selectedLanguage = detectedLanguage;
        browser.storage.local.set({ selectedLanguage });
      }
    }
  });

  browser.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "toggleCaptureButtons") {
      toggleCaptureButtons(false);
      browser.storage.local.set({ capturingState: { isCapturing: false } })
        .catch(function(error) {
          console.error("Error storing capturing state:", error);
        });
    }
  });
});
