// Smart Folder script: upload, auto-analyze, auto-rename, folders grid

// DOM elements
const dom = {
  detectMode: document.getElementById('detectMode'),
  externalConfig: document.getElementById('externalConfig'),
  apiUrl: document.getElementById('apiUrl'),
  apiKey: document.getElementById('apiKey'),
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('fileInput'),
  folderGrid: document.getElementById('folderGrid'),
  fabMain: document.getElementById('fabMain'),
  actionAdd: document.getElementById('actionAdd'),
  actionEdit: document.getElementById('actionEdit'),
  actionDelete: document.getElementById('actionDelete'),
  actionRandom: document.getElementById('actionRandom'),
  actionSwap: document.getElementById('actionSwap'),
  actionExport: document.getElementById('actionExport'),
  exportMenu: document.getElementById('exportMenu'),
  exportWith: document.getElementById('exportWith'),
  exportFlat: document.getElementById('exportFlat'),
};

// State
/** @typedef {{id:string,name:string,images:string[],expanded?:boolean}} Folder */
/** @typedef {{id:string,dataUrl:string,type:string,clothType?:string,colorName?:string,colorCode?:string,twFamily?:string,twShade?:number,uniqueId?:string,setId?:string,setIndex?:number,setTotal?:number,fileName?:string}} Image */

const state = {
  folders /** @type {Folder[]} */: [],
  images /** @type {Record<string, Image>} */: {},
  selectedFolders: new Set(),
  selectedImages: new Set(),
};

// Utils
function genId(){return Math.random().toString(36).slice(2,10)}
function sanitizeFilename(name){return name.replace(/[^a-z0-9_,\-\.\s]/gi,'').trim()}
function fileToDataURL(file){return new Promise((resolve,reject)=>{const r=new FileReader();r.onload=()=>resolve(r.result);r.onerror=reject;r.readAsDataURL(file);})}
function hexToRgb(hex){const h=hex.replace('#','');const i=parseInt(h,16);return{r:(i>>16)&255,g:(i>>8)&255,b:i&255}}
function rgbToHex(rgb){const toHex=n=>n.toString(16).padStart(2,'0');return `#${toHex(rgb.r)}${toHex(rgb.g)}${toHex(rgb.b)}`}
function srgbToLinear(c){c/=255;return c<=0.04045?c/12.92:Math.pow((c+0.055)/1.055,2.4)}
function rgbToLab({r,g,b}){
  const R=srgbToLinear(r),G=srgbToLinear(g),B=srgbToLinear(b);
  const X=R*0.4124+G*0.3576+B*0.1805;
  const Y=R*0.2126+G*0.7152+B*0.0722;
  const Z=R*0.0193+G*0.1192+B*0.9505;
  const Xn=0.95047,Yn=1.00000,Zn=1.08883;
  const fx = X/Xn>0.008856?Math.cbrt(X/Xn):(7.787*(X/Xn)+16/116);
  const fy = Y/Yn>0.008856?Math.cbrt(Y/Yn):(7.787*(Y/Yn)+16/116);
  const fz = Z/Zn>0.008856?Math.cbrt(Z/Zn):(7.787*(Z/Zn)+16/116);
  return {L:116*fy-16,a:500*(fx-fy),b:200*(fy-fz)};
}
function labDistance(a,b){const dL=a.L-b.L,da=a.a-b.a,db=a.b-b.b;return Math.sqrt(dL*dL+da*da+db*db)}
function rgbDistance(a,b){const dr=a.r-b.r,dg=a.g-b.g,db=a.b-b.b;return dr*dr+dg*dg+db*db}
function loadImage(src){return new Promise((res,rej)=>{const img=new Image();img.onload=()=>res(img);img.onerror=rej;img.src=src;})}
function createUniqueId(clothType){const first=(clothType||'x').trim().charAt(0).toUpperCase()||'X';const num=Math.floor(100000+Math.random()*900000);return `${first}${num}`}

// Tailwind palette subset
const tailwindPalette={
  slate:{50:'#f8fafc',100:'#f1f5f9',200:'#e2e8f0',300:'#cbd5e1',400:'#94a3b8',500:'#64748b',600:'#475569',700:'#334155',800:'#1e293b',900:'#0f172a'},
  gray:{50:'#f9fafb',100:'#f3f4f6',200:'#e5e7eb',300:'#d1d5db',400:'#9ca3af',500:'#6b7280',600:'#4b5563',700:'#374151',800:'#1f2937',900:'#111827'},
  zinc:{50:'#fafafa',100:'#f4f4f5',200:'#e4e4e7',300:'#d4d4d8',400:'#a1a1aa',500:'#71717a',600:'#52525b',700:'#3f3f46',800:'#27272a',900:'#18181b'},
  neutral:{50:'#fafafa',100:'#f5f5f5',200:'#e5e5e5',300:'#d4d4d4',400:'#a3a3a3',500:'#737373',600:'#525252',700:'#404040',800:'#262626',900:'#171717'},
  stone:{50:'#fafaf9',100:'#f5f5f4',200:'#e7e5e4',300:'#d6d3d1',400:'#a8a29e',500:'#78716c',600:'#57534e',700:'#44403c',800:'#292524',900:'#1c1917'},
  red:{50:'#fef2f2',100:'#fee2e2',200:'#fecaca',300:'#fca5a5',400:'#f87171',500:'#ef4444',600:'#dc2626',700:'#b91c1c',800:'#991b1b',900:'#7f1d1d'},
  orange:{50:'#fff7ed',100:'#ffedd5',200:'#fed7aa',300:'#fdba74',400:'#fb923c',500:'#f97316',600:'#ea580c',700:'#c2410c',800:'#9a3412',900:'#7c2d12'},
  amber:{50:'#fffbeb',100:'#fef3c7',200:'#fde68a',300:'#fcd34d',400:'#fbbf24',500:'#f59e0b',600:'#d97706',700:'#b45309',800:'#92400e',900:'#78350f'},
  yellow:{50:'#fefce8',100:'#fef9c3',200:'#fef08a',300:'#fde047',400:'#facc15',500:'#eab308',600:'#ca8a04',700:'#a16207',800:'#854d0e',900:'#713f12'},
  lime:{50:'#f7fee7',100:'#ecfccb',200:'#d9f99d',300:'#bef264',400:'#a3e635',500:'#84cc16',600:'#65a30d',700:'#4d7c0f',800:'#3f6212',900:'#365314'},
  green:{50:'#f0fdf4',100:'#dcfce7',200:'#bbf7d0',300:'#86efac',400:'#4ade80',500:'#22c55e',600:'#16a34a',700:'#15803d',800:'#166534',900:'#14532d'},
  emerald:{50:'#ecfdf5',100:'#d1fae5',200:'#a7f3d0',300:'#6ee7b7',400:'#34d399',500:'#10b981',600:'#059669',700:'#047857',800:'#065f46',900:'#064e3b'},
  teal:{50:'#f0fdfa',100:'#ccfbf1',200:'#99f6e4',300:'#5eead4',400:'#2dd4bf',500:'#14b8a6',600:'#0d9488',700:'#0f766e',800:'#115e59',900:'#134e4a'},
  cyan:{50:'#ecfeff',100:'#cffafe',200:'#a5f3fc',300:'#67e8f9',400:'#22d3ee',500:'#06b6d4',600:'#0891b2',700:'#0e7490',800:'#155e75',900:'#164e63'},
  sky:{50:'#f0f9ff',100:'#e0f2fe',200:'#bae6fd',300:'#7dd3fc',400:'#38bdf8',500:'#0ea5e9',600:'#0284c7',700:'#0369a1',800:'#075985',900:'#0c4a6e'},
  blue:{50:'#eff6ff',100:'#dbeafe',200:'#bfdbfe',300:'#93c5fd',400:'#60a5fa',500:'#3b82f6',600:'#2563eb',700:'#1d4ed8',800:'#1e40af',900:'#1e3a8a'},
  indigo:{50:'#eef2ff',100:'#e0e7ff',200:'#c7d2fe',300:'#a5b4fc',400:'#818cf8',500:'#6366f1',600:'#4f46e5',700:'#4338ca',800:'#3730a3',900:'#312e81'},
  violet:{50:'#f5f3ff',100:'#ede9fe',200:'#ddd6fe',300:'#c4b5fd',400:'#a78bfa',500:'#8b5cf6',600:'#7c3aed',700:'#6d28d9',800:'#5b21b6',900:'#4c1d95'},
  purple:{50:'#faf5ff',100:'#f3e8ff',200:'#e9d5ff',300:'#d8b4fe',400:'#c084fc',500:'#a855f7',600:'#9333ea',700:'#7e22ce',800:'#6b21a8',900:'#581c87'},
  fuchsia:{50:'#fdf4ff',100:'#fae8ff',200:'#f5d0fe',300:'#f0abfc',400:'#e879f9',500:'#d946ef',600:'#c026d3',700:'#a21caf',800:'#86198f',900:'#701a75'},
  pink:{50:'#fdf2f8',100:'#fce7f3',200:'#fbcfe8',300:'#f9a8d4',400:'#f472b6',500:'#ec4899',600:'#db2777',700:'#be185d',800:'#9d174d',900:'#831843'},
  rose:{50:'#fff1f2',100:'#ffe4e6',200:'#fecdd3',300:'#fda4af',400:'#fb7185',500:'#f43f5e',600:'#e11d48',700:'#be123c',800:'#9f1239',900:'#881337'},
};
const shadeOrder=[50,100,200,300,400,500,600,700,800,900];
// Requested descriptive shade names mapping
const shadeNameMap={
  50:'lightest',
  100:'lighter',
  200:'light',
  300:'soft',
  400:'base',
  500:'primary',
  600:'semi-dark',
  700:'dark',
  800:'darker',
  900:'deepest'
};
function friendlyShadeName(family,shade){
  const shadeName=shadeNameMap[shade]||'primary';
  const fam=String(family).toLowerCase();
  return `${fam} ${shadeName}`;
}
function generateAllShadeCombos(family){
  const fam=String(family).toLowerCase();
  const combos=shadeOrder.map(s=>`${fam} ${shadeNameMap[s]}`);
  return combos.join(', ');
}
function nearestTailwindColor(rgb){
  const sampleLab=rgbToLab(rgb);
  let best={dist:Infinity,family:'',shade:500,hex:'#000000',name:''};
  for(const [family,shades] of Object.entries(tailwindPalette)){
    for(const shade of shadeOrder){
      const hex=shades[shade];
      const c=hexToRgb(hex);
      const lab=rgbToLab(c);
      const d=labDistance(sampleLab,lab);
      if(d<best.dist){best={dist:d,family,shade,hex,name:friendlyShadeName(family,shade)};}
    }
  }
  return best;
}
async function getDominantColor(dataUrl){
  const img=await loadImage(dataUrl);
  const canvas=document.createElement('canvas');
  const ctx=canvas.getContext('2d');
  const size=128;canvas.width=size;canvas.height=size;ctx.drawImage(img,0,0,size,size);
  const {data}=ctx.getImageData(0,0,size,size);
  const start=Math.floor(size*0.2),end=Math.floor(size*0.8);
  const samples=[];
  function rgbToHsv(r,g,b){r/=255;g/=255;b/=255;const max=Math.max(r,g,b),min=Math.min(r,g,b);let h=0,s=0,v=max;const d=max-min;s=max===0?0:d/max;if(max===min)h=0;else{switch(max){case r:h=(g-b)/d+(g<b?6:0);break;case g:h=(b-r)/d+2;break;case b:h=(r-g)/d+4;break;}h/=6;}return{h,s,v};}
  for(let y=start;y<end;y++){
    for(let x=start;x<end;x++){
      const i=(y*size+x)*4;const a=data[i+3];if(a<80)continue;const r=data[i],g=data[i+1],b=data[i+2];
      const {s,v}=rgbToHsv(r,g,b); if(v>0.97||v<0.08) continue; if(s<0.15) continue; // skip near white/black and very desaturated
      samples.push({r,g,b});
    }
  }
  if(samples.length<50){
    let r=0,g=0,b=0,count=0;
    for(let y=start;y<end;y++){
      for(let x=start;x<end;x++){
        const i=(y*size+x)*4;const a=data[i+3];if(a<10)continue;r+=data[i];g+=data[i+1];b+=data[i+2];count++;
      }
    }
    if(!count) return {r:127,g:127,b:127};
    return {r:Math.round(r/count),g:Math.round(g/count),b:Math.round(b/count)};
  }
  // k-means (K=3)
  const K=3; const centers=[];
  for(let k=0;k<K;k++){centers.push({...samples[(Math.random()*samples.length)|0]});}
  for(let iter=0;iter<6;iter++){
    const groups=Array.from({length:K},()=>({sumR:0,sumG:0,sumB:0,count:0}));
    for(const p of samples){
      let bi=0,bd=Infinity;for(let k=0;k<K;k++){const d=rgbDistance(p,centers[k]);if(d<bd){bd=d;bi=k;}}
      const g=groups[bi];g.sumR+=p.r;g.sumG+=p.g;g.sumB+=p.b;g.count++;
    }
    for(let k=0;k<K;k++){const g=groups[k];if(g.count>0){centers[k]={r:Math.round(g.sumR/g.count),g:Math.round(g.sumG/g.count),b:Math.round(g.sumB/g.count)};}}
  }
  const counts=centers.map(()=>0);
  for(const p of samples){let bi=0,bd=Infinity;for(let k=0;k<K;k++){const d=rgbDistance(p,centers[k]);if(d<bd){bd=d;bi=k;}}counts[bi]++;}
  let maxI=0;for(let k=1;k<K;k++){if(counts[k]>counts[maxI]) maxI=k;}
  return centers[maxI];
}

// MobileNet clothing suggestion
let mobilenetModelPromise=null;
function loadScript(src){return new Promise((resolve,reject)=>{const s=document.createElement('script');s.src=src;s.async=true;s.onload=resolve;s.onerror=reject;document.head.appendChild(s);})}
async function getMobileNetModel(){if(!mobilenetModelPromise){mobilenetModelPromise=(async()=>{await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js');await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.1.0'); // @ts-ignore
const model=await mobilenet.load({version:2,alpha:1.0});return model;})()}return mobilenetModelPromise}
function mapLabelToClothType(label){const l=label.toLowerCase();
  if(/(t[- ]?shirt|tee|shirt)/.test(l))return 't_shirt';
  if(/(hoodie|sweatshirt)/.test(l))return 'hoodie';
  if(/(jeans|denim)/.test(l))return 'jeans';
  if(/(jacket|coat|parka|trench)/.test(l))return 'jacket';
  if(/(dress|gown)/.test(l)){if(/gown/.test(l))return 'long_gown';if(/cocktail/.test(l))return 'cocktail_dress';return 'dress';}
  if(/(suit|tuxedo|blazer)/.test(l))return 'tuxedo';
  if(/(skirt)/.test(l))return 'skirt';
  if(/(shorts)/.test(l))return 'shorts';
  if(/(sweater|pullover|cardigan)/.test(l))return 'sweater';
  if(/(hat|cap|beanie)/.test(l))return 'hat';
  return 'clothing';
}
async function predictClothType(model,dataUrl){const img=await loadImage(dataUrl); // @ts-ignore
  const preds=await model.classify(img);const mapped=preds.map(p=>({type:mapLabelToClothType(p.className),prob:p.probability})).filter(p=>p.type).sort((a,b)=>b.prob-a.prob);return mapped.length?mapped[0].type:'clothing'}

// Upload handlers
function attachUploadHandlers(){
  dom.fileInput.addEventListener('change', async()=>{
    const files=Array.from(dom.fileInput.files||[]).filter(f=>f.type.startsWith('image/'));
    dom.fileInput.value='';
    if(!files.length) return;
    await ingestFiles(files);
  });
  ['dragenter','dragover'].forEach(evt=>dom.dropzone.addEventListener(evt,e=>{e.preventDefault();dom.dropzone.classList.add('dragover');}));
  ['dragleave','drop'].forEach(evt=>dom.dropzone.addEventListener(evt,e=>{e.preventDefault();dom.dropzone.classList.remove('dragover');}));
  dom.dropzone.addEventListener('drop', async(e)=>{
    const dt=e.dataTransfer;const files=Array.from(dt?.files||[]).filter(f=>f.type.startsWith('image/'));
    if(!files.length) return;
    await ingestFiles(files);
  });
}

async function ingestFiles(files){
  const newImages=[];
  for(const file of files){
    const dataUrl=await fileToDataURL(file);
    const id=genId();
    state.images[id]={id,dataUrl,type:file.type};
    newImages.push(state.images[id]);
  }
  const model=await getMobileNetModel();
  for(const img of newImages){
    const avg=await getDominantColor(img.dataUrl);
    const detectedHex=rgbToHex(avg);
    const nearest=nearestTailwindColor(avg);
    img.colorCode=detectedHex; // apply detected dominant color to picker
    img.colorName=nearest.name; img.twFamily=nearest.family; img.twShade=nearest.shade;
    img.colorShadesList=generateAllShadeCombos(nearest.family);
    img.clothType=await predictClothType(model,img.dataUrl);
  }
  // Group by clothType + colorName to assign Unique ID and numbering
  const groups=new Map();
  for(const img of newImages){
    const key=`${img.clothType}__${img.colorName}`;
    if(!groups.has(key)) groups.set(key,[]);
    groups.get(key).push(img);
  }
  for(const [,arr] of groups){
    const uid=createUniqueId(arr[0].clothType);
    arr.forEach((img,idx)=>{img.uniqueId=uid; img.setId=uid; img.setIndex=idx+1; img.setTotal=arr.length;});
  }
  // Compose file names and place into cloth-type folders
  for(const img of newImages){
    const typePart=img.clothType; const shadePart=(img.colorName||'').toLowerCase().replace(/\s+/g,'_');
    const suffix=` ${img.setIndex} of ${img.setTotal}`;
    img.fileName=`${img.uniqueId}, ${typePart}_${shadePart}${suffix}`;
    // Folder by clothType
    let folder=state.folders.find(f=>f.name===img.clothType);
    if(!folder){folder={id:genId(),name:img.clothType,images:[]};state.folders.push(folder);}    
    folder.images.push(img.id);
  }
  render();
}

// Rendering with row-first placement across responsive columns
function getColumnsCount(){
  const w=window.innerWidth;
  if (w >= 1024) return 4; // Desktop bigger folders
  if (w >= 640) return 2;  // Tablet bigger folders
  return 2;                // Mobile unchanged
}

function render(){
  dom.folderGrid.innerHTML='';
  const colsCount=getColumnsCount();
  const columns=[];
  for(let i=0;i<colsCount;i++){const col=document.createElement('div');col.className='column';columns.push(col);dom.folderGrid.appendChild(col);}  
  state.folders.forEach((folder, idx)=>{
    const colIdx = idx % colsCount;
    const card=document.createElement('div');card.className='folder'+(state.selectedFolders.has(folder.id)?' selected':'');
    const header=document.createElement('div');header.className='folder-header';
    const title=document.createElement('div');title.className='folder-title';title.textContent=folder.name;
    const count=document.createElement('div');count.className='item-count';count.textContent=`${folder.images.length} items`;
    header.appendChild(title);header.appendChild(count);
    const content=document.createElement('div');content.className='folder-content';
    if (folder.expanded) content.classList.add('expanded');
    const inner=document.createElement('div');inner.className='content-inner';
    // allow dropping onto folder content to move images here
    inner.addEventListener('dragover',(e)=>{e.preventDefault();});
    inner.addEventListener('drop',(e)=>{e.preventDefault();const draggedId=e.dataTransfer?.getData('text/plain');if(!draggedId) return; moveImageToFolder(draggedId, folder.id);});
    folder.images.forEach(imgId=>{
      const img=state.images[imgId];
      const cell=document.createElement('div');
      cell.className='file-cell';
      const thumb=document.createElement('div');thumb.className='thumb'+(state.selectedImages.has(img.id)?' selected':'');
      thumb.draggable=true;
      thumb.addEventListener('dragstart',(e)=>{e.dataTransfer?.setData('text/plain', img.id);thumb.classList.add('dragging');});
      thumb.addEventListener('dragend',()=>{thumb.classList.remove('dragging');});
      const image=document.createElement('img');image.src=img.dataUrl;thumb.appendChild(image);
      const meta=document.createElement('div');meta.className='file-meta';
      const name=document.createElement('div');name.className='name';name.textContent=`${img.fileName}`;
      const picker=document.createElement('input');picker.type='color';picker.className='color-picker';picker.value=img.colorCode||'#000000';
      picker.addEventListener('input',(e)=>{
        const hex=picker.value;
        img.colorCode=hex;
        const nearest=nearestTailwindColor(hexToRgb(hex));
        img.twFamily=nearest.family; img.twShade=nearest.shade; img.colorName=nearest.name;
        img.colorShadesList=generateAllShadeCombos(nearest.family);
        // update filename and recompute grouping
        recomputeSetIndicesAndNames();
        render();
      });
      meta.appendChild(name);meta.appendChild(picker);
      cell.appendChild(thumb);cell.appendChild(meta);
      thumb.addEventListener('click',(e)=>{const multi=e.ctrlKey||e.metaKey;if(!multi)state.selectedImages.clear();if(state.selectedImages.has(img.id))state.selectedImages.delete(img.id);else state.selectedImages.add(img.id); render();});
      inner.appendChild(cell);
    });
    content.appendChild(inner);
    header.addEventListener('click',(e)=>{
      const multi=e.ctrlKey||e.metaKey;
      if(!multi) state.selectedFolders.clear();
      if(state.selectedFolders.has(folder.id)) state.selectedFolders.delete(folder.id); else state.selectedFolders.add(folder.id);
      folder.expanded = !folder.expanded;
      render();
    });
    card.appendChild(header);card.appendChild(content);
    columns[colIdx].appendChild(card);
  });
}

// Actions
dom.actionAdd.addEventListener('click',()=>{const name=prompt('New folder name');if(!name)return;state.folders.push({id:genId(),name,images:[]});render();});
dom.actionEdit.addEventListener('click',()=>{if(state.selectedFolders.size!==1){alert('Select exactly one folder to rename.');return;}const fid=[...state.selectedFolders][0];const folder=state.folders.find(f=>f.id===fid);const name=prompt('Folder name',folder?.name||'');if(!name||!folder)return;folder.name=name;render();});
dom.actionDelete.addEventListener('click',()=>{if(state.selectedFolders.size<1){alert('Select folder(s) to delete.');return;}const ids=new Set(state.selectedFolders);state.folders=state.folders.filter(f=>!ids.has(f.id));render();});
dom.actionRandom.addEventListener('click',()=>{if(state.selectedImages.size<1){alert('Select image(s) to assign new Unique ID.');return;}const uid=createUniqueId('x');state.selectedImages.forEach(id=>{const img=state.images[id];img.uniqueId=uid;const typePart=img.clothType;const shadePart=(img.colorName||'').replace(/\s+/g,'_');const suffix=` ${img.setIndex||1} of ${img.setTotal||1}`;img.fileName=`${img.uniqueId}, ${typePart}_${shadePart}${suffix}`;});render();});
dom.actionSwap.addEventListener('click',()=>{if(state.selectedFolders.size!==2){alert('Select exactly two folders to swap.');return;}const [aId,bId]=[...state.selectedFolders];const ai=state.folders.findIndex(f=>f.id===aId);const bi=state.folders.findIndex(f=>f.id===bId);if(ai<0||bi<0)return;const tmp=state.folders[ai];state.folders[ai]=state.folders[bi];state.folders[bi]=tmp;render();});
dom.actionExport.addEventListener('click',()=>{dom.exportMenu.classList.toggle('show');});
dom.exportWith.addEventListener('click',()=>{downloadImages(true)});
dom.exportFlat.addEventListener('click',()=>{downloadImages(false)});

function downloadImages(withFolders){
  const imgs=Object.values(state.images);
  for(const img of imgs){
    const folderName=(state.folders.find(f=>f.images.includes(img.id))?.name)||'';
    const name=withFolders && folderName? `${folderName}__${img.fileName}`: img.fileName;
    const a=document.createElement('a');a.href=img.dataUrl;a.download=sanitizeFilename(name);document.body.appendChild(a);a.click();a.remove();
  }
}

function moveImageToFolder(imgId,targetFolderId){
  const targetFolder=state.folders.find(f=>f.id===targetFolderId);
  if(!targetFolder) return;
  const sourceFolder=state.folders.find(f=>f.images.includes(imgId));
  if(sourceFolder){sourceFolder.images=sourceFolder.images.filter(id=>id!==imgId);} 
  targetFolder.images.push(imgId);
  recomputeSetIndicesAndNames();
  render();
}

function recomputeSetIndicesAndNames(){
  const groups=new Map();
  for(const img of Object.values(state.images)){
    const key=`${img.uniqueId||''}__${img.clothType||''}__${img.colorName||''}`;
    if(!groups.has(key)) groups.set(key,[]);
    groups.get(key).push(img);
  }
  for(const [,list] of groups){
    const ordered=[];
    for(const folder of state.folders){
      for(const id of folder.images){
        const img=state.images[id];
        if(list.includes(img)) ordered.push(img);
      }
    }
    const total=ordered.length||1;
    ordered.forEach((img,idx)=>{
      img.setTotal=total; img.setIndex=idx+1;
      const typePart=img.clothType; const shadePart=(img.colorName||'').toLowerCase().replace(/\s+/g,'_'); const suffix=` ${img.setIndex} of ${img.setTotal}`;
      img.fileName=`${img.uniqueId}, ${typePart}_${shadePart}${suffix}`;
    });
  }
}

// Settings toggle
dom.detectMode.addEventListener('change',()=>{const v=dom.detectMode.value;dom.externalConfig.classList.toggle('hidden', v!=='external');});

// Init
attachUploadHandlers();
render();
window.addEventListener('resize',()=>{render()});