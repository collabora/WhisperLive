importScripts("/../encoders/Mp3Encoder.min.js");

let NUM_CH = 2, // constant
    sampleRate = 16000,
    options = undefined,
    maxBuffers = undefined,
    encoder = undefined,
    recBuffers = undefined,
    socket = undefined,
    bufferCount = 0;

function error(message) {
  self.postMessage({ command: "error", message: "mp3: " + message });
}

function init(data) {
  if (data.config.numChannels === NUM_CH) {
    sampleRate = data.config.sampleRate;
    options = data.options;
  } else
    error("numChannels must be " + NUM_CH);
};

function setOptions(opt) {
  if (encoder || recBuffers)
    error("cannot set options during recording");
  else
    options = opt;
}

function start(bufferSize) {
  maxBuffers = Math.ceil(options.timeLimit * sampleRate / bufferSize);
  // TODO: get server address from user
  socket = new WebSocket("ws://localhost:9090/");
  socket.onopen = function(e) { 
    socket.send("handshake");
  };
  socket.onmessage = (event) => {
    self.postMessage({ command: "transcription", text: event.data});
  };

  if (options.encodeAfterRecord)
    recBuffers = [];
  else
    encoder = new Mp3LameEncoder(sampleRate, options.mp3.bitRate);
}

function record(buffer) {
  if (bufferCount++ < maxBuffers){
    if (encoder)
      encoder.encode(buffer);
    else 
      recBuffers.push(buffer);
    
    // send buffer to server
    socket.send(buffer[0]);
  }
  else
    self.postMessage({ command: "timeout" });
};

function postProgress(progress) {
  self.postMessage({ command: "progress", progress: progress });
};

function finish() {
  if (recBuffers) {
    postProgress(0);
    encoder = new Mp3LameEncoder(sampleRate, options.mp3.bitRate);
    let timeout = Date.now() + options.progressInterval;
    while (recBuffers.length > 0) {
      encoder.encode(recBuffers.shift());
      let now = Date.now();
      if (now > timeout) {
        postProgress((bufferCount - recBuffers.length) / bufferCount);
        timeout = now + options.progressInterval;
      }
    }
    postProgress(1);
  }
  self.postMessage({
    command: "complete",
    blob: encoder.finish(options.mp3.mimeType)
  });
  cleanup();
};

function cleanup() {
  encoder = recBuffers = undefined;
  bufferCount = 0;
  socket.close();
}

self.onmessage = function(event) {
  let data = event.data;
  switch (data.command) {
    case "init":    init(data);                 break;
    case "options": setOptions(data.options);   break;
    case "start":   start(data.bufferSize);     break;
    case "record":  record(data.buffer);        break;
    case "finish":  finish();                   break;
    case "cancel":  cleanup();
  }
};

self.postMessage({ command: "loaded" });
