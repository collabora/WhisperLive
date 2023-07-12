

var elem_container = null;
var elem_text = null;

var segments = [];
var text_segments = [];

function init_element() {
    if (document.getElementById('transcription')) {
        return;
    }

    elem_container = document.createElement('div');
    elem_container.id = "transcription";
    elem_container.style.cssText = 'padding-top:16px;font-size:18px;line-height:18px;top:0px;position:absolute;width:500px;height:90px;opacity:0.9;z-index:100;background:black;border-radius:10px;color:white;';

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

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    const { type, data } = request;
    
    if (type === "STOP") {
        remove_element();
        sendResponse({data: "STOPPED"});
        return;
    }

    init_element();

    message = JSON.parse(data);

    var text = '';
    for (var i = 0; i < message.length; i++) {
        text += message[i].text + ' ';
    }
    text = text.replace(/(\r\n|\n|\r)/gm, "");
    
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
});
