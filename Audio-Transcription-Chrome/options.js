/**
 * Captures audio from the active tab in Google Chrome.
 * @returns {Promise<MediaStream>} A promise that resolves with the captured audio stream.
 */
function captureTabAudio() {
  return new Promise((resolve) => {
    chrome.tabCapture.capture(
      {
        audio: true,
        video: false,
      },
      (stream) => {
        resolve(stream);
      }
    );
  });
}


/**
 * Sends a message to a specific tab in Google Chrome.
 * @param {number} tabId - The ID of the tab to send the message to.
 * @param {any} data - The data to be sent as the message.
 * @returns {Promise<any>} A promise that resolves with the response from the tab.
 */
function sendMessageToTab(tabId, data) {
  return new Promise((resolve) => {
    chrome.tabs.sendMessage(tabId, data, (response) => {
      resolve(response);
    });
  });
}

function generateUUID() {
  let dt = new Date().getTime();
  const uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = (dt + Math.random() * 16) % 16 | 0;
    dt = Math.floor(dt / 16);
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
  return uuid;
}

// Global variables for audio processing
let audioContext = null;
let preNode = null;
let socket = null;
let isServerReady = false;
let currentStream = null;
let currentOptions = null;

// AudioWorklet URL - make sure this path matches your manifest.json
const WORKLET_URL = chrome.runtime.getURL('audiopreprocessor.js');

async function initAudioWorklet(stream) {
  audioContext = new AudioContext();
  if (audioContext.state === 'suspended') {
    await audioContext.resume();
  }

  try {
    await audioContext.audioWorklet.addModule(WORKLET_URL);
    preNode = new AudioWorkletNode(audioContext, 'audiopreprocessor');
    const mediaStream = audioContext.createMediaStreamSource(stream);
    
    mediaStream.connect(preNode);
    preNode.connect(audioContext.destination);
    preNode.port.onmessage = (event) => {
      const data = event.data;
      
      
      const audio16k = data; // Float32Array @ 16 kHz
      
      if (socket && socket.readyState === WebSocket.OPEN && isServerReady) {
        socket.send(audio16k);
      }
    };
        
    // Test if we can hear audio (this will help verify the audio path)
    
  } catch (error) {
    console.error("Error initializing AudioWorklet:", error);
    throw error;
  }
}

function cleanupAudio() {
  
  if (preNode) {
    preNode.port.onmessage = null;
    preNode.disconnect();
    preNode = null;
  }
  
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }

  if (currentStream) {
    currentStream.getTracks().forEach(track => {
      track.stop();
      console.log("Stopped track:", track.kind);
    });
    currentStream = null;
  }
}

/**
 * Starts recording audio from the captured tab.
 * @param {Object} option - The options object containing the currentTabId.
 */
async function startRecord(option) {
  currentOptions = option;
  const stream = await captureTabAudio();
  const uuid = generateUUID();

  if (stream) {
    currentStream = stream;
    stream.oninactive = () => {
      cleanupAudio();
      window.close();
    };

    try {
      await initAudioWorklet(stream);
    } catch (error) {
      console.error("Failed to initialize AudioWorklet:", error);
      return;
    }

    socket = new WebSocket(`ws://${option.host}:${option.port}/`);
    isServerReady = false;
    let language = option.language;

    socket.onopen = function(e) {
      socket.send(
        JSON.stringify({
          uid: uuid,
          language: option.language,
          task: option.task,
          model: option.modelSize,
          use_vad: option.useVad
        })
      );
    };

    socket.onmessage = async (event) => {
      const data = JSON.parse(event.data);
      if (data["uid"] !== uuid)
        return;
      
      if (data["status"] === "WAIT"){
        await sendMessageToTab(option.currentTabId, {
          type: "showWaitPopup",
          data: data["message"],
        });
        chrome.runtime.sendMessage({ action: "toggleCaptureButtons", data: false }) 
        chrome.runtime.sendMessage({ action: "stopCapture" })
        return;
      }
        
      if (isServerReady === false){
        isServerReady = true;
        return;
      }
      
      if (language === null) {
        language = data["language"];
        
        // send message to popup.js to update dropdown
        chrome.runtime.sendMessage({
          action: "updateSelectedLanguage",
          detectedLanguage: language,
        });

        return;
      }

      if (data["message"] === "DISCONNECT"){
        chrome.runtime.sendMessage({ action: "toggleCaptureButtons", data: false, saveCaptions: option.saveCaptions });        
        return;
      }

      const res = await sendMessageToTab(option.currentTabId, {
        type: "transcript",
        data: {
          data: event.data,
          saveCaptions: option.saveCaptions,
        },
      });
    };

    socket.onclose = () => {
      cleanupAudio();
    };

    socket.onerror = (error) => {
      cleanupAudio();
    };

  } else {
    window.close();
  }
}


/**
 * Listener for incoming messages from the extension's background script.
 * @param {Object} request - The message request object.
 * @param {Object} sender - The sender object containing information about the message sender.
 * @param {Function} sendResponse - The function to send a response back to the message sender.
 */
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  const { type, data } = request;

  switch (type) {
    case "start_capture":
      startRecord(data);
      break;
    default:
      break;
  }

  sendResponse({});
  return true;
});
