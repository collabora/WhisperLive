importScripts("/../encoders/WavEncoder.min.js");


let sampleRate = 16000,
    numChannels = 2,
    options = undefined,
    maxBuffers = undefined,
    encoder = undefined,
    recBuffers = undefined,
    socket = undefined,
    bufferCount = 0;

function error(message) {
  self.postMessage({ command: "error", message: "wav: " + message });
}

function init(data) {
  sampleRate = data.config.sampleRate;
  numChannels = data.config.numChannels;
  options = data.options;
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
    console.log(event.data);
  };
  
  if (options.encodeAfterRecord)
    recBuffers = [];
  else
    encoder = new WavAudioEncoder(sampleRate, numChannels);
}

function record(buffer) {
  if (bufferCount++ < maxBuffers){
    if (encoder) 
      encoder.encode(buffer);
    else if(recBuffers)
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
    encoder = new WavAudioEncoder(sampleRate, numChannels);
    var timeout = Date.now() + options.progressInterval;
    while (recBuffers.length > 0) {
      encoder.encode(recBuffers.shift());
      var now = Date.now();
      if (now > timeout) {
        postProgress((bufferCount - recBuffers.length) / bufferCount);
        timeout = now + options.progressInterval;
      }
    }
    postProgress(1);
  }
  self.postMessage({
    command: "complete",
    blob: encoder.finish(options.wav.mimeType)
  });
  cleanup();
};

function cleanup() {
  encoder = recBuffers = undefined;
  bufferCount = 0;
  socket.close();
}

self.onmessage = function(event) {
  var data = event.data;
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
