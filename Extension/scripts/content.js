const captionObserver = new MutationObserver(newCaptions)

function startObserver(){
    const captionContainer = document.querySelector(".ytp-caption-window-container")
    if (captionContainer){
        captionObserver.observe(captionContainer,{childList: true, subtree: true})
        console.log("Observer started on youtube caption container")
    }else{
        console.log("Cannot find the container, trying again in 1 second")
        setTimeout(startObserver, 1000);
    }
}

function newCaptions(mutations, observer){
    mutations.forEach(mutation => {
        mutation.addedNodes.forEach(node => {
            if (node.nodeType == Node.TEXT_NODE){
                console.log(node.textContent);  
                translateCaption(node.textContent);
            }
        })
    });
}

async function translateCaption(caption){
    try {
        let response = await fetch(`http://127.0.0.1:8000/translation/${caption}`)

        if (!response.ok){
            throw new Error("Response was an error: ${response.status}");
        }
        
        let translatedCaption = await response.json();

        console.log(translatedCaption);
    }
    catch(error){
        console.error("Error connecting to the API", error)
    }
}

setTimeout(startObserver, 20000)