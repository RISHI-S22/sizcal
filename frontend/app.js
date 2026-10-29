const SCAN_DURATION=7000;
const PRE_SCAN_DELAY=1200;
const FRAME_INTERVAL=250;

const positions=[
 {key:"front",title:"FRONT VIEW",display:"Front View",instruction:"Face the camera directly. Keep your entire body visible.",nextInstruction:"Turn 90° to your LEFT."},
 {key:"left",title:"LEFT SIDE VIEW",display:"Left Side View",instruction:"Turn 90° to your left. Keep your entire body visible.",nextInstruction:"Turn completely around so your BACK faces the camera."},
 {key:"back",title:"BACK VIEW",display:"Back View",instruction:"Turn completely around. Keep your entire body visible.",nextInstruction:"Turn 90° to your RIGHT."},
 {key:"right",title:"RIGHT SIDE VIEW",display:"Right Side View",instruction:"Turn 90° to your right. Keep your entire body visible.",nextInstruction:""}
];

let currentIndex=0;
let cameraStream=null;
let scanning=false;
let lastResult=null;
let selectedCategory = "top_wear";
let capturedViews={front:null,left:null,back:null,right:null};
let bestFrame=null;
let bestScore=-Infinity;
let lastFrameTime=0;

const video=document.getElementById("cameraVideo");
const overlay=document.getElementById("cameraOverlay");
const canvas=document.getElementById("cameraCanvas");
const startScanBtn=document.getElementById("startScanBtn");
const stopScanBtn=document.getElementById("stopScanBtn");

document.addEventListener("DOMContentLoaded",async()=>{
 await checkHealth();
 updatePosition();
 startScanBtn?.addEventListener("click",startBodyScan);
 stopScanBtn?.addEventListener("click",stopBodyScan);
 document.getElementById("saveMeasurementBtn")?.addEventListener("click",saveMeasurement);
 document.getElementById("newScanBtn")?.addEventListener("click",()=>{resetResults();startBodyScan()});
 document.getElementById("clearHistoryBtn")?.addEventListener("click",clearHistory);
 await loadHistory();
});
const topWearBtn =
    document.getElementById(
        "topWearBtn"
    );

const bottomWearBtn =
    document.getElementById(
        "bottomWearBtn"
    );

topWearBtn.addEventListener(
    "click",
    () => {
        selectCategory("top_wear");
    }
);

bottomWearBtn.addEventListener(
    "click",
    () => {
        selectCategory("bottom_wear");
    }
);
function selectCategory(category) {

    if (
        category !== "top_wear" &&
        category !== "bottom_wear"
    ) {
        return;
    }

    selectedCategory = category;

    document
        .getElementById(
            "clothingCategory"
        )
        .value = category;

    const topButton =
        document.getElementById(
            "topWearBtn"
        );

    const bottomButton =
        document.getElementById(
            "bottomWearBtn"
        );

    topButton.classList.toggle(
        "active",
        category === "top_wear"
    );

    bottomButton.classList.toggle(
        "active",
        category === "bottom_wear"
    );

    const title =
        document.getElementById(
            "cameraCenterMessage"
        );

    if (title) {

        if (category === "top_wear") {

            title.textContent =
                "TOP WEAR SCAN";

        } else {

            title.textContent =
                "BOTTOM WEAR SCAN";

        }
    }

    console.log(
        "Selected category:",
        category
    );
}

async function checkHealth(){
 const status=document.getElementById("apiStatus");
 try{
  const r=await fetch("/api/health",{cache:"no-store"});
  const d=await r.json();
  status.textContent=d.success?"SYSTEM READY":"API ERROR";
  if(d.success)status.classList.add("online");
 }catch(e){
  status.textContent="API OFFLINE";
 }
}


async function startCamera(){
 if(cameraStream)return true;
 if(!navigator.mediaDevices?.getUserMedia){
  alert("Camera access is not available in this browser.");return false;
 }
 try{
  cameraStream=await navigator.mediaDevices.getUserMedia({
   video:{width:{ideal:1280},height:{ideal:720},facingMode:"user"},
   audio:false
  });
  video.srcObject=cameraStream;
  video.muted=true;video.autoplay=true;video.playsInline=true;
  await video.play();
  resizeOverlay();drawGuide();
  return true;
 }catch(e){
  console.error(e);
  if(e.name==="NotAllowedError")alert("Camera permission was denied. Allow camera access and try again.");
  else if(e.name==="NotFoundError")alert("No camera was found.");
  else if(e.name==="NotReadableError")alert("The camera is already being used by another application.");
  else alert("Unable to access the camera.");
  cameraStream=null;return false;
 }
}

async function startBodyScan(){
    const heightInput =
    document.getElementById(
        "userHeight"
    );

const userHeight =
    Number(
        heightInput.value
    );

if (
    !Number.isFinite(userHeight) ||
    userHeight < 100 ||
    userHeight > 250
) {

    alert(
        "Please enter your actual height between 100 and 250 cm."
    );

    heightInput.focus();

    return;
}

if (
    selectedCategory !== "top_wear" &&
    selectedCategory !== "bottom_wear"
) {

    alert(
        "Please select Top Wear or Bottom Wear."
    );

    return;
}
 if(scanning)return;
 const height=Number(document.getElementById("userHeight")?.value);
 if(!height||height<100||height>250){
  alert("Please enter your height between 100 cm and 250 cm.");return;
 }
 if(!(await startCamera()))return;

 scanning=true;currentIndex=0;lastResult=null;
 capturedViews={front:null,left:null,back:null,right:null};
 resetResults();startScanBtn.hidden=true;stopScanBtn.hidden=false;
 document.getElementById("cameraCenterMessage").style.display="none";
 await scanCurrentPosition();
}

async function scanCurrentPosition(){
 if(!scanning)return;
 const p=positions[currentIndex];
 updatePosition();setScanState("GET READY");
 await wait(PRE_SCAN_DELAY);
 if(!scanning)return;

 setScanState("SCANNING");
 document.getElementById("scanLine")?.classList.add("active");
 bestFrame=null;bestScore=-Infinity;lastFrameTime=0;

 await collectBestFrame(performance.now());

 if(!scanning)return;
 document.getElementById("scanLine")?.classList.remove("active");

 if(!bestFrame){
  alert(`Could not capture the ${p.display} frame. Please try again.`);
  stopBodyScan();return;
 }

 capturedViews[p.key]=bestFrame;
 markPositionComplete(currentIndex);

 if(currentIndex<positions.length-1){
  setScanState("TURN");
  const box=document.getElementById("cameraCenterMessage");
  box.innerHTML=`<div class="message-icon">↻</div><strong>${escapeHtml(p.nextInstruction)}</strong><span>Next scan starts automatically.</span>`;
  box.style.display="flex";
  await wait(2500);
  if(!scanning)return;
  box.style.display="none";
  currentIndex++;
  await scanCurrentPosition();
 }else{
  await processAllViews();
 }
}

function collectBestFrame(startTime){
 return new Promise(resolve=>{
  const loop=async now=>{
   if(!scanning){resolve();return}
   const elapsed=now-startTime;
   if(elapsed>=SCAN_DURATION){resolve();return}
   if(now-lastFrameTime>=FRAME_INTERVAL){
    lastFrameTime=now;
    await evaluateFrame();
   }
   updateTimer(Math.max(0,SCAN_DURATION-elapsed));
   requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
 });
}

async function evaluateFrame(){
 if(!video.videoWidth||!video.videoHeight)return;

 const small=document.createElement("canvas");
 small.width=240;small.height=160;
 const ctx=small.getContext("2d",{willReadFrequently:true});
 ctx.drawImage(video,0,0,240,160);
 const data=ctx.getImageData(0,0,240,160);
 const score=frameScore(data);

 if(score<=bestScore)return;

 canvas.width=video.videoWidth;canvas.height=video.videoHeight;
 const full=canvas.getContext("2d");
 full.drawImage(video,0,0,canvas.width,canvas.height);
 const blob=await toBlob(canvas,"image/jpeg",.90);
 if(blob){bestScore=score;bestFrame=blob}
}

function frameScore(imageData){
 const d=imageData.data;let sum=0,sq=0,edge=0,n=0;
 const w=imageData.width,h=imageData.height;
 for(let y=1;y<h;y+=3)for(let x=1;x<w;x+=3){
  const i=(y*w+x)*4;
  const g=.299*d[i]+.587*d[i+1]+.114*d[i+2];
  sum+=g;sq+=g*g;
  const j=(y*w+x-1)*4;
  const gl=.299*d[j]+.587*d[j+1]+.114*d[j+2];
  edge+=Math.abs(g-gl);n++;
 }
 const mean=sum/n;
 const contrast=Math.sqrt(Math.max(0,sq/n-mean*mean));
 return edge/n*.65+contrast*.35-Math.abs(mean-128)*.15;
}

async function processAllViews(){
 setScanState("ANALYZING");
 const height=Number(document.getElementById("userHeight")?.value);
 const form=new FormData();

 for(const key of ["front","left","back","right"]){
  if(!capturedViews[key]){alert("All four views were not captured.");stopBodyScan();return}
  form.append(`${key}_image`,capturedViews[key],`${key}.jpg`);
 }

 form.append("user_height_cm",String(height));

 try{
  const r=await fetch("/api/camera/measure-multiview",{method:"POST",body:form});
  const d=await r.json();
  if(!r.ok||!d.success)throw new Error(typeof d.detail==="string"?d.detail:JSON.stringify(d.detail||d));
  lastResult=d;displayResults(d);
  scanning=false;stopCamera();startScanBtn.hidden=false;stopScanBtn.hidden=true;setScanState("COMPLETE");
 }catch(e){
  console.error(e);alert("Body scan failed:\n"+e.message);stopBodyScan();
 }
}

const recommendation =
    data.recommendation;

if (
    recommendation &&
    recommendation.available === false
) {

    setText(
        "recommendedSize",
        "Coming Soon"
    );

    setText(
        "confidenceValue",
        "Bottom-wear models are being prepared."
    );

    return;
}

if (recommendation) {

    setText(
        "recommendedSize",
        recommendation.recommended_size
    );

    setText(
        "confidenceValue",
        `${recommendation.confidence_percent}% confidence`
    );

    displayProbabilities(
        recommendation.probabilities
    );

    displayBrandRecommendations(
        recommendation.brand_recommendations
    );
}
async function saveMeasurement(){
 if(!lastResult){alert("Complete a scan first.");return}
 try{
  const record={
   measurement:lastResult.measurement,
   recommendation:lastResult.recommendation,
   brand_recommendations:lastResult.brand_recommendations,
   user_height_cm:lastResult.user_height_cm,
   source:"camera-multiview"
  };
  const form=new FormData();form.append("data",JSON.stringify(record));
  const r=await fetch("/api/history/save",{method:"POST",body:form});
  const d=await r.json();
  if(!r.ok||!d.success)throw new Error(d.error||"Save failed");
  alert("Measurement saved successfully.");await loadHistory();
 }catch(e){alert("Unable to save measurement: "+e.message)}
}

async function loadHistory(){
 const list=document.getElementById("historyList");if(!list)return;
 try{
  const d=await (await fetch("/api/history",{cache:"no-store"})).json();
  const history=d.history||[];
  if(!history.length){list.innerHTML='<div class="empty-history">No saved measurements yet.</div>';return}
  list.innerHTML=history.slice(0,10).map(x=>{
   const rec=x.recommendation||{};
   return `<div class="history-item"><div><strong>${escapeHtml(rec.brand_name||"Brand")}</strong><br><small>Size ${escapeHtml(rec.recommended_size||"--")}</small></div><small>${escapeHtml(x.saved_at||"")}</small></div>`
  }).join("");
 }catch(e){console.error(e)}
}

async function clearHistory(){
 if(!confirm("Clear all saved measurements?"))return;
 await fetch("/api/history",{method:"DELETE"});await loadHistory();
}

function updatePosition(){
 const p=positions[currentIndex];
 setText("currentPosition",p.title);setText("positionTitle",p.display);setText("positionDescription",p.instruction);
 updateSteps();resetTimer();
}

function updateSteps(){
 ["scanStepFront","scanStepLeft","scanStepBack","scanStepRight"].forEach((id,i)=>{
  const e=document.getElementById(id);if(!e)return;
  e.classList.remove("active","complete");
  if(i<currentIndex)e.classList.add("complete");
  if(i===currentIndex)e.classList.add("active");
 });
}

function markPositionComplete(i){
 const ids=["scanStepFront","scanStepLeft","scanStepBack","scanStepRight"];
 document.getElementById(ids[i])?.classList.add("complete");
}

function setScanState(s){setText("scanState",s);setText("scanStatusText",s)}
function updateTimer(ms){
 const sec=Math.ceil(ms/1000);setText("timerText",String(sec));
 const p=Math.max(0,Math.min(1,ms/SCAN_DURATION));
 const c=2*Math.PI*44;
 const e=document.getElementById("timerProgress");if(e)e.style.strokeDashoffset=c*(1-p);
}
function resetTimer(){setText("timerText","7");const e=document.getElementById("timerProgress");if(e)e.style.strokeDashoffset=2*Math.PI*44}

function resizeOverlay(){
 if(!video.videoWidth||!overlay)return;
 overlay.width=video.videoWidth;overlay.height=video.videoHeight;
 overlay.style.width=video.clientWidth+"px";overlay.style.height=video.clientHeight+"px";
}
function drawGuide(){
 const c=overlay?.getContext("2d");if(c)c.clearRect(0,0,overlay.width,overlay.height);
}
function stopCamera(){
 cameraStream?.getTracks().forEach(t=>t.stop());cameraStream=null;
 if(video)video.srcObject=null;drawGuide();
}
function stopBodyScan(){
 scanning=false;stopCamera();startScanBtn.hidden=false;stopScanBtn.hidden=true;setScanState("READY");
 const box=document.getElementById("cameraCenterMessage");
 box.innerHTML='<div class="message-icon">◉</div><strong>Position yourself inside the frame</strong><span>Your scan will begin automatically</span>';box.style.display="flex";
 document.getElementById("scanLine")?.classList.remove("active");
}
function resetResults(){lastResult=null;document.getElementById("scanComplete").hidden=true}
function wait(ms){return new Promise(r=>setTimeout(r,ms))}
function toBlob(c,t,q){return new Promise(r=>c.toBlob(r,t,q))}
function formatCm(v){return v==null||Number.isNaN(Number(v))?"--":Number(v).toFixed(2)}
function setText(id,v){const e=document.getElementById(id);if(e)e.textContent=v}
function escapeHtml(v){return String(v).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#039;")}
window.addEventListener("resize",()=>{resizeOverlay();drawGuide()});
video?.addEventListener("loadedmetadata",()=>{resizeOverlay();drawGuide()});
