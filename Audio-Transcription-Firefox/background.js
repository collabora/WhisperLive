browser.runtime.onMessage.addListener(async function(request, sender, sendResponse) {
  const { action, data } = request;
  if (action === "transcript") {
    await browser.tabs.query({ active: true, currentWindow: true })
      .then((tabs) => {
        const tabId = tabs[0].id;
        browser.tabs.sendMessage(tabId, { action: "show_transcript", data });
      })
      .catch((error) => {
        console.error("Error retrieving active tab:", error);
      });
  }
  if (action === "updateSelectedLanguage") {
    const detectedLanguage = data;
    try {
      await browser.storage.local.set({ selectedLanguage: detectedLanguage });
      browser.tabs.query({ active: true, currentWindow: true }).then((tabs) => {
        const tabId = tabs[0].id;
        browser.tabs.sendMessage(tabId, { action: "updateSelectedLanguage", detectedLanguage });
      });
    } catch (error) {
      console.error("Error updateSelectedLanguage:", error);
    }
  }
  if (action === "toggleCaptureButtons") {    
    try {
      await browser.storage.local.set({ capturingState: { isCapturing: false } });
      browser.tabs.query({ active: true, currentWindow: true }).then((tabs) => {
        const tabId = tabs[0].id;
        browser.tabs.sendMessage(tabId, { action: "toggleCaptureButtons", data: false });
      });
    } catch (error) {
      console.error("Error updating capturing state:", error);
    }

    try{
      await browser.tabs.query({ active: true, currentWindow: true })
        .then((tabs) => {
          const tabId = tabs[0].id;
          browser.tabs.sendMessage(tabId, { action: "stopCapture", data });
        })
        .catch((error) => {
          console.error("Error retrieving active tab:", error);
        }); 
    } catch (error) {
      console.error(error);
    }
  }
  
  if (action === "showPopup") {
    try{
      await browser.tabs.query({ active: true, currentWindow: true })
        .then((tabs) => {
          const tabId = tabs[0].id;
          browser.tabs.sendMessage(tabId, { action: "showWaitPopup", data });
        })
        .catch((error) => {
          console.error(error);
        });
    } catch (error) {
      console.error(error);
    }
  }
});

