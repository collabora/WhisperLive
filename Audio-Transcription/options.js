document.addEventListener('DOMContentLoaded', () => {
  const mute = document.getElementById('mute');
  const maxTime = document.getElementById('maxTime');
  const save = document.getElementById('save');
  const status = document.getElementById('status');
  const mp3Select = document.getElementById('mp3');
  const wavSelect = document.getElementById('wav');
  const quality = document.getElementById("quality");
  const qualityLi = document.getElementById("qualityLi");
  const limitRemoved = document.getElementById("removeLimit");
  const doVad = document.getElementById("doVad");
  let currentFormat;
  //initial settings
  chrome.storage.sync.get({
    muteTab: false,
    maxTime: 1200000,
    format: "mp3",
    quality: 192,
    limitRemoved: false,
    asr: false,
    doVad: false
  }, (options) => {
    mute.checked = options.muteTab;
    limitRemoved.checked = options.limitRemoved;
    maxTime.disabled = options.limitRemoved;
    maxTime.value = options.maxTime/60000;
    currentFormat = options.format;
    doVad.checked = options.doVad;
    if (options.format === "mp3") {
      mp3Select.checked = true;
      qualityLi.style.display = "block";
    } else {
      wavSelect.checked = true;
    }
    if (options.quality === "96") {
      quality.selectedIndex = 0;
    } else if(options.quality === "192") {
      quality.selectedIndex = 1;
    } else {
      quality.selectedIndex = 2;
    }
  });

  mute.onchange = () => {
    status.innerHTML = "";
  }

  doVad.onchange = () => {
    status.innerHTML = "";
  }

  maxTime.onchange = () => {
    status.innerHTML = "";
    if(maxTime.value > 20) {
      maxTime.value = 20;
    } else if (maxTime.value < 1) {
      maxTime.value = 1;
    } else if (isNaN(maxTime.value)) {
      maxTime.value = 20;
    }
  }

  mp3Select.onclick = () => {
    currentFormat = "mp3";
    qualityLi.style.display = "block";
    status.innerHTML = "";
  }

  wavSelect.onclick = () => {
    currentFormat = "wav";
    qualityLi.style.display = "none";
    status.innerHTML = "";
  }

  quality.onchange = (e) => {
    status.innerHTML = "";
  }

  limitRemoved.onchange = () => {
    if(limitRemoved.checked) {
      maxTime.disabled = true;
      status.innerHTML = "WARNING: Recordings that are too long may not save properly!"
    } else {
      maxTime.disabled = false;
      status.innerHTML = "";
    }
  }

  save.onclick = () => {
    chrome.storage.sync.set({
      muteTab: mute.checked,
      maxTime: maxTime.value*60000,
      format: currentFormat,
      quality: quality.value,
      limitRemoved: limitRemoved.checked,
      doVad: doVad.checked
    });
    status.innerHTML = "Settings saved!"
  }
});
