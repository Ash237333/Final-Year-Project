function change_captions(caption){
    const captionContainer = document.querySelector("#display_captions")
    if (captionContainer == null){
        throw new Error("Cannot find the caption container for the sidebar");
    }
    const captionElement = document.createElement("div");
    captionElement.textContent = caption;
    captionElement.style.padding = "8px 0";
    captionElement.style.borderBottom = "1px solid #ddd"; 

    captionContainer.prepend(captionElement);

    captionContainer.scrollTop = 0;
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "updateCaption") {
        change_captions(message.message)
    }
});