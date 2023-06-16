browser.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  const { action, data } = request;
  if (action === "transcript") {
    browser.tabs.query({ active: true, currentWindow: true })
      .then((tabs) => {
        const tabId = tabs[0].id;
        browser.tabs.sendMessage(tabId, { action: "show_transcript", data });
      })
      .catch((error) => {
        console.error("Error retrieving active tab:", error);
      });
  }
});




  
