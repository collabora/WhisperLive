

var elem_container = null;
var elem_text = null;

var segments = [];
var text_segments = [];

var lang = 'ru';

function initPopupElement() {
  if (document.getElementById('popupElement')) {
    return;
  }

  const popupContainer = document.createElement('div');
  popupContainer.id = 'popupElement';
  popupContainer.style.cssText = 'position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: white; color: black; padding: 16px; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5); display: none; text-align: center;';

  const popupText = document.createElement('span');
  popupText.textContent = 'Default Text';
  popupText.className = 'popupText';
  popupText.style.fontSize = '24px';
  popupContainer.appendChild(popupText);

  const buttonContainer = document.createElement('div');
  buttonContainer.style.marginTop = '8px';
  const closePopupButton = document.createElement('button');
  closePopupButton.textContent = 'Close';
  closePopupButton.style.backgroundColor = '#65428A';
  closePopupButton.style.color = 'white';
  closePopupButton.style.border = 'none';
  closePopupButton.style.padding = '8px 16px'; // Add padding for better click area
  closePopupButton.style.cursor = 'pointer';
  closePopupButton.addEventListener('click', async () => {
    popupContainer.style.display = 'none';
    await browser.runtime.sendMessage({ action: 'toggleCaptureButtons', data: false });
  });
  buttonContainer.appendChild(closePopupButton);
  popupContainer.appendChild(buttonContainer);

  document.body.appendChild(popupContainer);
}

function showPopup(customText) {
  const popup = document.getElementById('popupElement');
  const popupText = popup.querySelector('.popupText');

  if (popup && popupText) {
      popupText.textContent = customText || 'Default Text'; // Set default text if custom text is not provided
      popup.style.display = 'block';
  }
}

function resizeTranscription({w = 500, h = 90, f = 18, l = 18}){
    const transcriptionStyle = document.getElementById("transcription").style;
    transcriptionStyle.width = `${w}px`;
    transcriptionStyle.height = `${h}px`;
    transcriptionStyle.fontSize = `${f}px`;
    transcriptionStyle.lineHeight = `${l}px`;
}

var translateToggle = true;

function getLocalStorageValue(key) {
    return new Promise((resolve) => {
        chrome.storage.local.get([key], (result) => {
        resolve(result[key]);
        });
    });
}

function sendMessageToTab(tabId, data) {
    return new Promise((resolve) => {
        chrome.tabs.sendMessage(tabId, data, (response) => {
        resolve(response);
        });
    });
}

async function stopCapture() {
    const optionTabId = await getLocalStorageValue("optionTabId");
    const currentTabId = await getLocalStorageValue("currentTabId");

    if (optionTabId) {
        res = await sendMessageToTab(currentTabId, {
            type: "STOP",
            data: { currentTabId: currentTabId },
        });
    }
}

// const selectedLanguageTo = await getLocalStorageValue("selectedLanguageTo");

function speakText(text) {

    getLocalStorageValue("selectedLanguageTo")
        .then(lang => {
            console.log('selected lang', lang)
            const curLang = lang ?? 'ru'
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.voice = window.speechSynthesis.getVoices().filter(voice => {
                // return voice.lang.includes('en'.toLocaleLowerCase())
                const cond = voice.lang.toLocaleLowerCase().includes(curLang.toLocaleLowerCase());
                // cond && console.log('voice.lang', curLang, voice.lang)
                return cond;
            })[0];//.find(voice => voice.voiceURI === voiceInEl.value);

            // window.speechSynthesis.getVoices().forEach(voice => {
            //     console.log('voice')
            //     console.log(voice)
            // });

            utterance.pitch = 1;
            utterance.rate = 1;
            utterance.volume = 1;
            
            // speak that utterance
            console.log('translateToggle', translateToggle)
            translateToggle && window.speechSynthesis.speak(utterance);
        })
        /*
    // stop any speaking in progress
    // if (translateToggle){
    //     translateToggle = false;
    //     window.speechSynthesis.cancel();
    //     return;
    // }

    // translateToggle = true;
  
    // create new utterance with all the properties
    // const text = 'я@ты@я@ты@я@ты@я@ты@я@ты@';
    console.log('selected lang', lang)
    const curLang = lang ?? 'ru'
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.voice = window.speechSynthesis.getVoices().filter(voice => {
        // return voice.lang.includes('en'.toLocaleLowerCase())
        const cond = voice.lang.toLocaleLowerCase().includes(curLang.toLocaleLowerCase());
        cond && console.log('voice.lang', curLang, voice.lang)
        return cond;
    })[0];//.find(voice => voice.voiceURI === voiceInEl.value);

    // window.speechSynthesis.getVoices().forEach(voice => {
    //     console.log('voice')
    //     console.log(voice)
    // });

    utterance.pitch = 1;
    utterance.rate = 1;
    utterance.volume = 1;
    
    // speak that utterance
    window.speechSynthesis.speak(utterance);

    */

}

function speechChunks(){
    
    if (chunksToSpeech.length){
        if (window.speechSynthesis.speaking){
            setTimeout(() => speechChunks(), 100);
            return;
        }
        const nc = [];
        // chunksTotal
        chunksToSpeech.forEach(chunk => {
            if (!chunksTotal.includes(chunk)){
                nc.push(chunk);
            }
        })

        chunksToSpeech = [...chunksToSpeech, ...nc]
        console.log('nc', nc)

        const text = nc.join('. ');
        console.log('speechChunks', text)
        // chunksToSpeech = [];
        speakText(text);
    }    
}

var chunkSeq = [];

function speechChunksNew(chunks, tmp){
    chunkSeq.push({chunks, tmp});
    speechChunksGo();
}

function speechChunksGo(){
    // if (chunks.length === 0) return;
    // console.log('speechChunksNew', window.speechSynthesis.speaking, tmp, chunks)
    if (window.speechSynthesis.speaking){
        setTimeout(() => speechChunksGo(), 500);
        // setTimeout(() => speechChunksNew(chunks, tmp), 500);
        return;
    }
    if (!chunkSeq) return;

    const {chunks, tmp} = chunkSeq.shift();
    const text = chunks.map(s => s.trim()).join(' ').trim();
    console.log('text', tmp, text)

    // window.speechSynthesis.getVoices().forEach(voice => {
    //     console.log('voice')
    //     console.log(voice)
    // });
    
    speakText(text);
    
}

// updateVoices()

function updateVoices() {
    // add an option for each available voice that isn't already added
    // window.speechSynthesis.getVoices().forEach(voice => {
    //   const isAlreadyAdded = [...voiceInEl.options].some(option => option.value === voice.voiceURI);
    //   if (!isAlreadyAdded) {
    //     const option = new Option(voice.name, voice.voiceURI, voice.default, voice.default);
    //     voiceInEl.add(option);
    //   }
    // });

    window.speechSynthesis.getVoices().forEach(voice => {
        console.log('voice')
        console.log(voice)
        // SpeechSynthesisVoice {voiceURI: 'Лаура', name: 'Лаура', lang: 'sk-SK', localService: true, default: false}
    });
}

document.addEventListener('keydown', function(event) {
    // Проверяем, что нажата клавиша Control и клавиша "1"
    if (event.ctrlKey && event.key === '1') {
        resizeTranscription({w: 500, h: 90, f: 18, l: 18})
    }

    if (event.ctrlKey && event.key === '2') {
        resizeTranscription({w: 750, h: 135, f: 27, l: 27})
    }

    if (event.ctrlKey && event.key === '3') {
        resizeTranscription({w: 1000, h: 180, f: 36, l: 36})
    }

    if (event.ctrlKey && event.key.toLowerCase() === 'a') {
        // speakText();
        translateToggle = !translateToggle;
        if (!translateToggle){
            window.speechSynthesis.cancel();
        } else {
            speechChunksGo();
        }
        console.log('translateToggle', translateToggle)
    }


    event.preventDefault();
});

function init_element() {
    if (document.getElementById('transcription')) {
        return;
    }

    elem_container = document.createElement('div');
    elem_container.id = "transcription";
    elem_container.style.cssText = 'padding-top:16px;font-size:18px;position: fixed; top: 85%; left: 50%; transform: translate(-50%, -50%);line-height:18px;width:500px;height:90px;opacity:0.9;z-index:100;background:black;border-radius:10px;color:white;';

    for (var i = 0; i < 4; i++) {
        elem_text = document.createElement('span');
        elem_text.style.cssText = 'position: absolute;padding-left:16px;padding-right:16px;';
        elem_text.id = "t" + i;
        elem_container.appendChild(elem_text);

        if (i == 3) {
            elem_text.style.top = "-1000px"
        }
    }

    document.body.appendChild(elem_container);

    let x = 0;
    let y = 0;

    // Query the element
    const ele = elem_container;

    // Handle the mousedown event
    // that's triggered when user drags the element
    const mouseDownHandler = function (e) {
        // Get the current mouse position
        x = e.clientX;
        y = e.clientY;

        // Attach the listeners to `document`
        document.addEventListener('mousemove', mouseMoveHandler);
        document.addEventListener('mouseup', mouseUpHandler);
    };

    const mouseMoveHandler = function (e) {
        // How far the mouse has been moved
        const dx = e.clientX - x;
        const dy = e.clientY - y;

        // Set the position of element
        ele.style.top = `${ele.offsetTop + dy}px`;
        ele.style.left = `${ele.offsetLeft + dx}px`;

        // Reassign the position of mouse
        x = e.clientX;
        y = e.clientY;
    };

    const mouseUpHandler = function () {
        // Remove the handlers of `mousemove` and `mouseup`
        document.removeEventListener('mousemove', mouseMoveHandler);
        document.removeEventListener('mouseup', mouseUpHandler);
    };

    ele.addEventListener('mousedown', mouseDownHandler);
}

function getStyle(el,styleProp)
{
    var x = document.getElementById(el);
    if (x.currentStyle)
        var y = x.currentStyle[styleProp];
    else if (window.getComputedStyle)
        var y = document.defaultView.getComputedStyle(x,null).getPropertyValue(styleProp);
    return y;
}

function get_lines(elem, line_height) {
    var divHeight = elem.offsetHeight;
    var lines = divHeight / line_height;

    var original_text = elem.innerHTML;

    var words = original_text.split(' ');
    var segments = [];
    var current_lines = 1;
    var segment = '';
    var segment_len = 0;
    for (var i = 0; i < words.length; i++)
    {
        segment += words[i] + ' ';
        elem.innerHTML = segment;
        divHeight = elem.offsetHeight;

        if ((divHeight / line_height) > current_lines) {
            var line_segment = segment.substring(segment_len, segment.length - 1 - words[i].length - 1);
            segments.push(line_segment);
            segment_len += line_segment.length + 1;
            current_lines++;
        }
    }

    var line_segment = segment.substring(segment_len, segment.length - 1)
    segments.push(line_segment);

    elem.innerHTML = original_text;

    return segments;

}

function remove_element() {
    var elem = document.getElementById('transcription')
    for (var i = 0; i < 4; i++) {
        document.getElementById("t" + i).remove();
    }
    elem.remove()
}

var chunksOld = [];
var chunksToSpeech = [];
var chunksTotal = [];

class SentenceProcessor {
    constructor() {
        this.buffer = ''; // Буфер для неполных данных
        this.completeSentencesSet = new Set(); // Множество для уникальных предложений
        this.prevChunks = []; // Хранение предыдущих обработанных кусков
    }

    processChunks(currentChunks) {
        // Найдем пересечение предыдущих кусков и текущих
        const overlap = this.prevChunks.filter(chunk => currentChunks.includes(chunk)).join(' ');

        // Определяем начало новой части данных, которая может содержать новые предложения
        const newPartStartIndex = overlap ? currentChunks.indexOf(overlap.split(' ')[0]) + overlap.split(' ').length : 0;
        const newPart = currentChunks.slice(newPartStartIndex).join(' ');

        // Обновляем буфер новыми данными
        this.buffer += newPart;

        // Шаблон для поиска полного предложения, которое заканчивается на точку и пробел
        const sentencePattern = /([^.!?]+[.!?] )/g;
        let match;
        let newSentences = [];

        while ((match = sentencePattern.exec(this.buffer)) !== null) {
            const sentence = match[1].trim();
            // Если предложение новое, добавляем его в результат и набор уже известных предложений
            if (!this.completeSentencesSet.has(sentence)) {
                this.completeSentencesSet.add(sentence);
                newSentences.push(sentence);
            }
        }

        // Оставляем в буфере только ту часть, которая может содержать незавершенные предложения
        this.buffer = this.buffer.slice(sentencePattern.lastIndex);

        // Обновляем предыдущие куски
        this.prevChunks = currentChunks;

        return newSentences;
    }
}

var processor = new SentenceProcessor();

// const chunks1 = ["Это ", "первое ", "предложение. ", "Второе ", "предложение ", "начинается "];
// const chunks2 = ["предложение. ", "Третье ", "начинается ", "здесь. ", "Четвертое ", "предложение "];
// const chunks3 = ["здесь. ", "Четвертое ", "предложение ", "оканчивается. "];

// console.log(processor.processChunks(chunks1)); // ["Это первое предложение."]
// console.log(processor.processChunks(chunks2)); // ["Второе предложение начинается здесь."]
// console.log(processor.processChunks(chunks3)); // ["Третье предложение заканчивается."]

var superChunks = [];
var cnt = 0;

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    const { type, data } = request;
    console.log('chrome.runtime.onMessage.addListener', request)
    lang = data.lang;
    if (type === "STOP") {
        remove_element();
        sendResponse({data: "STOPPED"});
        return true;
    } else if (type === "showWaitPopup"){
        initPopupElement();

        showPopup(`Estimated wait time ~ ${Math.round(data)} minutes`);
        sendResponse({data: "popup"});
        return true;
    } else if (type === "CHANGE_LANG"){
        lang = data.lang;
        console.log('set lang', lang)
        return true;
    }

    init_element();

    message = JSON.parse(data);
    message = message["segments"];
    console.log('message', message)

    // const ch = message.map(msg => msg.text)
    // console.log('processChunks', processor.processChunks(ch));

    var text = '';
    for (var i = 0; i < message.length; i++) {
        text += message[i].text + ' ';
    }
    text = text.replace(/(\r\n|\n|\r)/gm, "");
    console.log('text?', text);
    // const chunks = text.split('. ')
    const chunks = text.split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(Boolean);


    // let chunks = text.match( /[^\.!\?]+[\.!\?]+/g );
    
    console.log('chunks?', chunks);
    // chunks = chunks.filter(Boolean).map(s => s.trim())

    let chunksNew = [];
    console.log('chunks.length', chunks.length, chunks)
    if (chunks.length > 3){
        if (superChunks.length === 0){
            superChunks = chunks.slice(0, -2);
            cnt = superChunks.length;
            if (superChunks.length !== 0){
                speechChunksNew(superChunks, 'superChunks')
            }
        } else {
            let varChunks = chunks.slice(1, -2)
            let newSuper  = [];
            varChunks.forEach(ch => {
                if (!superChunks.includes(ch)){
                    newSuper.push(ch);
                }
            })

            superChunks = [...superChunks, ...newSuper]
            // console.log('varChunks', varChunks)
            if (newSuper.length !== 0){                
                speechChunksNew(newSuper, 'newSuper')
            }
        }        
    } 

    console.log('superChunks', superChunks.join('. ').trim() + '.')

    if (chunksOld.length === 0){
        chunksOld = chunks;
    } else {
        chunks.forEach(chunk => {
            if (chunksOld.includes(chunk)){
                chunksNew.push(chunk);
            }
        })
        chunksOld = [...chunks];
        let chunkDiff = chunksNew.filter(Boolean).filter(chunk => !chunksToSpeech.includes(chunk)).filter(Boolean)
        // console.log('chunksNew', chunksNew)
        // console.log('chunkDiff', chunkDiff)
        chunksToSpeech = [...chunksToSpeech, ...chunkDiff];
    //    speechChunks();
    }
    
    var elem = document.getElementById('t3');
    elem.innerHTML = text;

    var line_height_style = getStyle('t3', 'line-height');
    var line_height = parseInt(line_height_style.substring(0, line_height_style.length - 2));
    var divHeight = elem.offsetHeight;
    var lines = divHeight / line_height;

    text_segments = [];
    text_segments = get_lines(elem, line_height);
    
    elem.innerHTML = '';

    if (text_segments.length > 2) {
        for (var i = 0; i < 3; i++) {
            document.getElementById('t' + i).innerHTML = text_segments[text_segments.length - 3 + i];
        }
    } else {
        for (var i = 0; i < 3; i++) {
            document.getElementById('t' + i).innerHTML = '';
        }
    }

    if (text_segments.length <= 2) {
        for (var i = 0; i < text_segments.length; i++) {
            document.getElementById('t' + i).innerHTML = text_segments[i];
        }
    } else {
        for (var i = 0; i < 3; i++) {
            document.getElementById('t' + i).innerHTML = text_segments[text_segments.length - 3 + i];
        }
    }

    for (var i = 1; i < 3; i++)
    {
        var parent_elem = document.getElementById('t' + (i - 1));
        var elem = document.getElementById('t' + i);
        elem.style.top = parent_elem.offsetHeight + parent_elem.offsetTop + 'px';
    }

    sendResponse({});
    return true;
});
