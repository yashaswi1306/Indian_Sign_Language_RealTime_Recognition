const interval=500;
const confidence_threshhold=30;

const vid=document.getElementById("vid")
const canvas=document.getElementById("canvas")
const statusText=document.getElementById("status_text")
const handFlag=document.getElementById("hand_flag")
const top3span=document.querySelectorAll("#top_list span")

navigator.mediaDevices.getUserMedia({
    video:true
}).then(stream => {
    vid.srcObject=stream;
    vid.onloadedmetadata=() => {
        canvas.width=vid.videoWidth; // get height and width from vidoes. turn 'coneccting.. to live'
        canvas.height=vid.videoHeight;
        statusText.textContent="Live";
        setInterval(predict,interval) // run the predict function at interval of 500ms
    };
});

async function predict() {
    const context=canvas.getContext("2d"); // html pe rendering ke liye. and then draw image draws the actual image from the vid onto the html element
    context.drawImage(vid,0,0);
    const image=canvas.toDataURL('image/jpeg',0.8) // 80% clarity

    //To send to backend:

    const res= await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type" : "application/json" },
        body: JSON.stringify({image}),
    });

    const data = await res.json(); // await for the predicted output from model
    updateUI(data);
}

function updateUI({label,confidence,top3}){

    let no_hands=false;

    if(!label || label==="unk")
    {
        no_hands=true;
    }

    handFlag.classList.toggle("hidden",!no_hands); // if no hands is false, then remove the 'no hand found'
    top3span.forEach((span,i)=> {
        // loops per item in top 3
        const item = top3 && top3[i] 
        if (item) // not null
        {
            span.textContent= `${item.label} : ${item.confidence.toFixed(0)}%`; // cuz confidence default 0%
        }
        else
        {
            span.textContent= `---`;
        }
    });

    if (no_hands||confidence<confidence_threshhold)
    {
        if(no_hands)
        {
            statusText.textContent= `No Hands Detected`;
        }
        else
        {
            statusText.textContent= `${label} : (low confidence)`;
        }

        return;
    }
}
