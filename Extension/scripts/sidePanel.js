function change_captions(caption){
    const captionContainer = document.querySelector("#display_captions")
    if (captionContainer == null){
        throw new Error("Cannot find the caption container for the sidebar");
    }
    captionContainer.innerHTML += `<br>${caption}`;
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "updateCaption") {
        change_captions(message.message)
    }
});