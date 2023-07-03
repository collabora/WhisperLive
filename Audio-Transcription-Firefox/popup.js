document.addEventListener("DOMContentLoaded", function() {
  var startButton = document.getElementById("startCapture");
  var stopButton = document.getElementById("stopCapture");

  startButton.addEventListener("click", function() {
    browser.tabs.query({ active: true, currentWindow: true })
      .then(function(tabs) {
        browser.tabs.sendMessage(tabs[0].id, { action: "startCapture" });
        toggleCaptureButtons(true);
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
            console.log(response);
            toggleCaptureButtons(false);
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
  }
});