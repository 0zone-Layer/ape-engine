import React, { useState, useEffect, useMemo, useCallback } from "react";

const M={
  mod:n=>((Math.round(n)%100)+100)%100,
  d1:n=>Math.floor(Math.abs(Math.round(n))%100/10),
  d2:n=>Math.abs(Math.round(n))%10,
  ds:n=>{const v=Math.abs(Math.round(n))%100;return Math.floor(v/10)+(v%10);},
  rev:n=>{const s=String(M.mod(n)).padStart(2,"0");return parseInt(s[1]+s[0],10);},
  dr:n=>{let x=M.ds(Math.abs(Math.round(n))%100);while(x>=10)x=M.ds(x);return x;},
  dp:n=>M.d1(n)*M.d2(n),
  mean:a=>a.length?a.reduce((s,v)=>s+v,0)/a.length:0,
  cd:(a,b)=>Math.min(Math.abs(a-b),100-Math.abs(a-b)),
  near:(p,a,t)=>M.cd(p,a)<=(t||2),
  nearR:(p,a,regime)=>M.cd(p,a)<=(regime==="volatile"?4:regime==="flat"?1:2),
  std:a=>{if(a.length<2)return 0;const avg=M.mean(a);return Math.sqrt(a.reduce((s,v)=>s+(v-avg)**2,0)/(a.length-1));},
  median:a=>{const s=[...a].sort((x,y)=>x-y),m=Math.floor(s.length/2);return s.length%2?s[m]:(s[m-1]+s[m])/2;},
};
const pad2=n=>String(M.mod(n)).padStart(2,"0");
const COLS=["A","B","C","D"];
const ok=v=>v!==null&&v!==undefined&&!isNaN(v);
const getSeries=(col,data)=>data.map(r=>r[col]).filter(v=>ok(v));
const PERF_NOW=()=>typeof performance!=="undefined"&&performance.now?performance.now():Date.now();
const CORE_ALGO_PRIORITY=["Markov","DeepMarkov4","PatternMemBank","KNNWindow","Sticky","ValueCluster","FreqDecay","CrossLagSelf","DFTPeriod","EntropyAdapt","LocalModePredict","RecencyGravity"];
const MIN_CROSS_SIGNAL_WEIGHT=0.35;
const HEAVY_PRED_THRESHOLD_MS=18;
const MAX_HEAVY_STREAK=8;
const LIGHTWEIGHT_TRIGGER_STREAK=2;
const LIGHTWEIGHT_HISTORY_THRESHOLD=350;

// ── GLOBAL SERIES: merges ALL datasets sorted by date then row ──────────────
// This is the key to cross-period learning: algos are backtested on ALL historical
// data, not just the current dataset. 2025 + 2026 data together = much better calibration.
function getGlobalSeries(col,datasets){
  const totalLen=Object.values(datasets||{}).reduce((s,ds)=>s+(ds.rows?.length||0),0);
  const ck=col+"_"+totalLen+"_"+_TC._ver;
  if(_TC.gs[ck])return _TC.gs[ck];
  const allRows=[];
  Object.values(datasets||{}).forEach(ds=>{
    (ds.rows||[]).forEach(r=>{if(ok(r[col]))allRows.push({v:r[col],date:r.date||null,row:r.row});});
  });
  allRows.sort((a,b)=>{
    if(a.date&&b.date)return a.date.localeCompare(b.date);
    if(a.date)return -1;if(b.date)return 1;return a.row-b.row;
  });
  const seen=new Set();
  const result=allRows.filter(r=>{const k=(r.date||"_")+"|"+r.row;if(seen.has(k))return false;seen.add(k);return true;}).map(r=>r.v);
  _TC.gs[ck]=result;
  return result;
}
const CLR={A:"#a78bfa",B:"#34d399",C:"#fbbf24",D:"#f87171"};
// Cached algo count — avoids Object.keys(A) in every render
let ALGO_COUNT=0; // filled after A{} definition

// ══════════════════════════════════════════════════════════════════
// ── MODULE-LEVEL TRAINING CACHE ──────────────────────────────────
// Lives outside React. During auto-train every expensive function
// (mineTransform, getSharedRowProps, getColGapSignals, getGlobalSeries)
// is called once per row. We cache by a cheap key (col+dataLen) so
// repeated calls within the same tick are O(1) lookups.
// Keys auto-invalidate because data length increments each row.
// ══════════════════════════════════════════════════════════════════
const _TC={
  tx:{},      // mineTransform:    "srcCol_tgtCol_len"
  sp:{},      // getSharedRowProps: "col_len"
  cg:{},      // getColGapSignals:  "col_len"
  gs:{},      // getGlobalSeries:   "col_totalLen"
  _ver:0,
  bumpVer(){this._ver=(this._ver+1)%1e9;},
  clear(){this.tx={};this.sp={};this.cg={};this.gs={};}
};

// ── BUILT-IN ALGORITHMS (count auto-tracked in ALGO_COUNT) ───────
const A={
  Reverse:        s=>[M.rev(s[s.length-1])],
  DigitSum:       s=>[M.mod(s[s.length-1]+M.ds(s[s.length-1]))],
  RevSumTf:       s=>[M.mod(s[s.length-1]+M.rev(s[s.length-1]))],
  MirrorDiff:     s=>{const v=s[s.length-1],d=Math.abs(M.d1(v)-M.d2(v));return[d?M.mod(d*11):M.mod(v+11)];},
  DigitalRoot:    s=>[M.mod(M.dr(s[s.length-1])*11)],
  Complement:     s=>[M.mod(100-s[s.length-1])],
  DigitProduct:   s=>{const v=s[s.length-1],dp=M.dp(v);return[M.mod(dp||v+1)];},
  RevComplement:  s=>[M.rev(M.mod(100-s[s.length-1]))],
  SumDoubled:     s=>[M.mod(s[s.length-1]*2)],
  DigSumChain:    s=>{const v=s[s.length-1];return[M.mod(v+M.ds(v+M.ds(v)))];},
  CubeDigit:      s=>{const v=s[s.length-1];return[M.mod(M.d1(v)**3+M.d2(v)**3)];},
  DigFact:        s=>{const f=[1,1,2,6,24,120,720,5040,40320,362880];const v=s[s.length-1];return[M.mod(f[M.d1(v)]+f[M.d2(v)])];},
  FibMod:         s=>{const v=s[s.length-1]%10;let a=0,b=1;for(let i=0;i<v;i++){const t=(a+b)%100;a=b;b=t;}return[b];},
  SqrtMod:        s=>[M.mod(Math.floor(Math.sqrt(s[s.length-1])*10))],
  TriNum:         s=>{const v=s[s.length-1]%13;return[M.mod(v*(v+1)/2)];},
  DigSumProd:     s=>{const v=s[s.length-1];return[M.mod(M.ds(v)*v)];},
  CollatzStep:    s=>{const v=s[s.length-1];return[M.mod(v%2===0?v/2:3*v+1)];},
  Mean3:          s=>[M.mod(Math.round(M.mean(s.slice(-3))))],
  Mean5:          s=>[M.mod(Math.round(M.mean(s.slice(-5))))],
  WtdMean:        s=>{const sl=s.slice(-6),tot=sl.reduce((a,_,i)=>a+Math.pow(2,i),0)||1;return[M.mod(Math.round(sl.reduce((a,v,i)=>a+v*Math.pow(2,i),0)/tot))];},
  Median5:        s=>{const sl=[...s.slice(-5)].sort((a,b)=>a-b),m=Math.floor(sl.length/2);return[M.mod(Math.round(sl.length%2?sl[m]:(sl[m-1]+sl[m])/2))];},
  HarmMean:       s=>{const sl=s.slice(-5).filter(v=>v>0);if(!sl.length)return[s[s.length-1]];return[M.mod(Math.round(sl.length/sl.reduce((a,v)=>a+1/v,0)))];},
  GeoMean:        s=>{const sl=s.slice(-5).filter(v=>v>0);if(!sl.length)return[s[s.length-1]];return[M.mod(Math.round(Math.pow(sl.reduce((a,v)=>a*v,1),1/sl.length)))];},
  MoveStd:        s=>{if(s.length<3)return[s[s.length-1]];const w=s.slice(-5),avg=M.mean(w),std=M.std(w);return[M.mod(Math.round(avg+std)),M.mod(Math.round(avg-std))];},
  ZScore:         s=>{if(s.length<4)return[s[s.length-1]];const avg=M.mean(s.slice(-8)),std=M.std(s.slice(-8));if(!std)return[s[s.length-1]];return[M.mod(Math.round(avg-(s[s.length-1]-avg)/std*std*0.5))];},
  ExpSmooth:      s=>{if(s.length<2)return[s[0]||0];const q=Math.max(1,Math.floor(s.length/4));let sm=M.mean(s.slice(0,q));s.slice(q).forEach(v=>{sm=0.3*v+0.7*sm;});return[M.mod(Math.round(sm))];},
  DblExp:         s=>{if(s.length<3)return[s[s.length-1]];const _q=Math.max(1,Math.floor(s.length/4));const _init=M.mean(s.slice(0,_q));let lv=_init,tr=(M.mean(s.slice(_q,_q*2))-_init)/(_q||1);for(let i=_q;i<s.length;i++){const pl=lv,pt=tr;lv=0.4*s[i]+0.6*(pl+pt);tr=0.3*(lv-pl)+0.7*pt;}return[M.mod(Math.round(lv+tr))];},
  KernelSmooth:   s=>{if(s.length<3)return[s[s.length-1]];const n=s.length,h=3;let ws=0,vs=0;for(let i=0;i<n;i++){const w=Math.exp(-0.5*((n-1-i)/h)**2);ws+=w;vs+=w*s[i];}return[M.mod(Math.round(vs/ws))];},
  MedianFilt:     s=>{if(s.length<3)return[s[s.length-1]];return[M.mod(Math.round(M.median(s.slice(-3))))];},
  LowPass:        s=>{if(s.length<2)return[s[0]||0];let sm=s[0];s.slice(1).forEach(v=>{sm=0.25*v+0.75*sm;});return[M.mod(Math.round(0.25*s[s.length-1]+0.75*sm))];},
  BandPass:       s=>{if(s.length<4)return[s[s.length-1]];const avg=M.mean(s.slice(-8)),std=M.std(s.slice(-8));const filt=s.filter(v=>Math.abs(v-avg)<=std);if(!filt.length)return[s[s.length-1]];return[M.mod(Math.round(M.mean(filt.slice(-4))))];},
  DiffFilt:       s=>{if(s.length<3)return[s[s.length-1]];const diffs=[];for(let i=1;i<s.length;i++){let d=s[i]-s[i-1];if(d>50)d-=100;if(d<-50)d+=100;diffs.push(d);}return[M.mod(s[s.length-1]+Math.round(M.mean(diffs.slice(-4))))];},
  AutoCorr:       s=>{if(s.length<6)return[s[s.length-1]];const n=s.length,avg=M.mean(s);
    // Fix: denominator uses same overlapping range as numerator (both i=lag..n)
    let bestLag=1,bestAcf=-2;
    for(let lag=1;lag<=Math.min(8,n-2);lag++){
      let num=0,den0=0,den1=0;
      for(let i=lag;i<n;i++){num+=(s[i]-avg)*(s[i-lag]-avg);den0+=(s[i]-avg)**2;den1+=(s[i-lag]-avg)**2;}
      const acf=(den0*den1)>0?num/Math.sqrt(den0*den1):0;
      if(Math.abs(acf)>Math.abs(bestAcf)){bestAcf=acf;bestLag=lag;}
    }
    return[M.mod(s[n-bestLag])];},
  WtdMomentum:    s=>{
    if(s.length<2)return[s[0]||0];
    let ws=0,wd=0;
    for(let i=1;i<s.length;i++){
      const w=Math.pow(1.8,i);
      let d=s[i]-s[i-1];if(d>50)d-=100;if(d<-50)d+=100;
      ws+=w;wd+=d*w;
    }
    const rawGap=wd/ws;
    // Cap extreme projections to ±25 to avoid overshooting
    const cappedGap=Math.max(-25,Math.min(25,Math.round(rawGap)));
    return[M.mod(s[s.length-1]+cappedGap)];
  },
  SecondDiff:     s=>{if(s.length<3)return[s[s.length-1]];const n=s.length;let d1=s[n-1]-s[n-2],d2=s[n-2]-s[n-3];if(d1>50)d1-=100;if(d1<-50)d1+=100;if(d2>50)d2-=100;if(d2<-50)d2+=100;return[M.mod(s[n-1]+(2*d1-d2))];},
  LastGap:        s=>{if(s.length<2)return[s[0]||0];let g=s[s.length-1]-s[s.length-2];if(g>50)g-=100;if(g<-50)g+=100;return[M.mod(s[s.length-1]+g)];},
  GapMedian:      s=>{if(s.length<3)return[s[s.length-1]];const gaps=[];for(let i=1;i<s.length;i++){let g=s[i]-s[i-1];if(g>50)g-=100;if(g<-50)g+=100;gaps.push(g);}const sg=[...gaps].sort((a,b)=>a-b),m=Math.floor(sg.length/2);return[M.mod(s[s.length-1]+Math.round(sg.length%2?sg[m]:(sg[m-1]+sg[m])/2))];},
  TheilSen:       s=>{
    if(s.length<4)return[s[s.length-1]];
    // Only use recent half for better responsiveness
    const recent=s.slice(-Math.max(4,Math.ceil(s.length/2)));
    const slopes=[];
    for(let i=0;i<recent.length-1;i++)for(let j=i+1;j<recent.length;j++){
      let sl=(recent[j]-recent[i])/(j-i);
      if(sl>50)sl-=100;if(sl<-50)sl+=100;
      slopes.push(sl);
    }
    slopes.sort((a,b)=>a-b);
    const med=slopes[Math.floor(slopes.length/2)];
    return[M.mod(s[s.length-1]+Math.round(med))];
  },
  LinFit:         s=>{const n=s.length;let sx=0,sy=0,sxy=0,sx2=0;s.forEach((v,i)=>{sx+=i;sy+=v;sxy+=i*v;sx2+=i*i;});const D=n*sx2-sx*sx;if(!D)return[s[n-1]];const a=(n*sxy-sx*sy)/D,b=(sy-a*sx)/n;return[M.mod(Math.round(a*n+b))];},
  QuadFit:        s=>{if(s.length<4)return[s[s.length-1]];try{const n=s.length;let sx=0,sx2=0,sx3=0,sx4=0,sy=0,sxy=0,sx2y=0;s.forEach((v,i)=>{sx+=i;sx2+=i*i;sx3+=i*i*i;sx4+=i*i*i*i;sy+=v;sxy+=i*v;sx2y+=i*i*v;});const det=n*(sx2*sx4-sx3*sx3)-sx*(sx*sx4-sx3*sx2)+sx2*(sx*sx3-sx2*sx2);if(!det)return[s[n-1]];const c0=(sy*(sx2*sx4-sx3*sx3)-sx*(sxy*sx4-sx3*sx2y)+sx2*(sxy*sx3-sx2*sx2y))/det;const c1=(n*(sxy*sx4-sx3*sx2y)-sy*(sx*sx4-sx3*sx2)+sx2*(sx*sx2y-sxy*sx2))/det;const c2=(n*(sx2*sx2y-sxy*sx3)-sx*(sx*sx2y-sxy*sx2)+sy*(sx*sx3-sx2*sx2))/det;return[M.mod(Math.round(c0+c1*n+c2*n*n))];}catch(e){return[s[s.length-1]];}},
  LCGFit:         s=>{
    if(s.length<3)return[s[s.length-1]];
    const n=s.length;let best={sc:-1,a:1,c:0};
    const perfect=n-1; // max achievable exact score
    outer:for(const a of[1,2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,61,71,73,79,83,89,97])
      for(const c of[0,1,3,5,7,11,13,17,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]){
        let sc=0;
        for(let i=1;i<n;i++){
          const pred=M.mod(a*s[i-1]+c);
          if(pred===s[i])sc+=1;
          else if(M.near(pred,s[i],1))sc+=0.3;
          // Early-abort: remaining steps can't beat best even if all exact
          else if(sc+(n-i)<best.sc)break;
        }
        if(sc>best.sc){best={sc,a,c};if(sc>=perfect)break outer;}
      }
    return[M.mod(best.a*s[n-1]+best.c)];
  },
  Recurrence2:    s=>{
    if(s.length<3)return[s[s.length-1]];
    const n=s.length;let best={sc:-1,pred:s[n-1]};
    const perfect=(n-2)*1.4; // approx max with recency weights
    outer2:for(const a of[1,2,3,-1,-2,5,-3])
      for(const b of[0,1,-1,2,-2,3])
        for(const c of[0,1,-1,3,-3,7,-7,11,-11,13,-13]){
          let sc=0;
          for(let i=2;i<n;i++){
            const p=M.mod(a*s[i-1]+b*s[i-2]+c);
            const rw=i>=n-5?1.5:1.0; // recency weight
            if(p===s[i])sc+=1.0*rw;
            else if(M.near(p,s[i],1))sc+=0.4*rw;
            else if(sc+(n-i)*1.5<best.sc)break;
          }
          if(sc>best.sc){best={sc,pred:M.mod(a*s[n-1]+b*s[n-2]+c)};if(sc>=perfect)break outer2;}
        }
    return[best.pred];
  },
  Cyclic:         s=>{
    if(s.length<4)return[s[s.length-1]];
    const n=s.length;let best={score:-1,pred:s[n-1],period:1};
    for(let p=2;p<=Math.min(14,Math.floor(n/2));p++){
      let sc=0;
      for(let i=p;i<n;i++)sc+=Math.max(0,1-M.cd(s[i],s[i-p])/12);
      const norm=sc/(n-p);
      if(norm>best.score){
        // Precision fix: instead of exact position, use weighted mean of all matching positions
        const positions=[];
        for(let i=n%p||p;i<=n;i+=p)if(i>0&&i<=n)positions.push(s[n-i]??s[n-1]);
        const wVals=positions.map((v,i)=>v*Math.exp(-i*0.3));
        const wSum=positions.map((_,i)=>Math.exp(-i*0.3)).reduce((a,b)=>a+b,1e-9);
        best={score:norm,pred:M.mod(Math.round(wVals.reduce((a,b)=>a+b,0)/wSum)),period:p};
      }
    }
    return[best.pred];
  },
  AR3:            s=>{
    if(s.length<4)return[s[s.length-1]];
    const n=s.length;let best={sc:-1,pred:s[n-1]};
    outer3:for(const a of[0.2,0.4,0.5,0.6,0.7,0.8,1.0,1.2])
      for(const b of[-0.4,-0.2,-0.1,0,0.1,0.2,0.4])
        for(const c of[-0.3,-0.1,0,0.1,0.3]){
          let sc=0;
          for(let i=3;i<n;i++){
            const p=M.mod(Math.round(a*s[i-1]+b*s[i-2]+c*s[i-3]));
            const rw=i>=n-5?1.5:1.0;
            if(p===s[i])sc+=1.0*rw;
            else if(M.near(p,s[i],2))sc+=0.3*rw;
            else if(sc+(n-i)*1.5<best.sc)break;
          }
          if(sc>best.sc){best={sc,pred:M.mod(Math.round(a*s[n-1]+b*s[n-2]+c*s[n-3]))};if(sc>=(n-3)*1.4)break outer3;}
        }
    return[best.pred];
  },
  MovReg:         s=>{const sl=s.slice(-6),n=sl.length;let sx=0,sy=0,sxy=0,sx2=0;sl.forEach((v,i)=>{sx+=i;sy+=v;sxy+=i*v;sx2+=i*i;});const D=n*sx2-sx*sx;if(!D)return[sl[n-1]];const a=(n*sxy-sx*sy)/D,b=(sy-a*sx)/n;return[M.mod(Math.round(a*n+b))];},
  LogMap:         s=>{if(s.length<3)return[s[s.length-1]];const x=s[s.length-1]/99;let bestR=3.5,bestSc=-1;for(let r=2.5;r<=3.99;r+=0.04){let sc=0,xr=s[0]/99;for(let i=1;i<s.length;i++){xr=r*xr*(1-xr);if(Math.abs(xr*99-s[i])<4)sc++;}if(sc>bestSc){bestSc=sc;bestR=r;if(sc>=s.length-2)break;}}return[M.mod(Math.round(bestR*(x)*(1-x)*99))];},
  PhaseNN:        s=>{if(s.length<6)return[s[s.length-1]];const n=s.length;let best={dist:Infinity,next:s[n-1]};for(let i=2;i<n-1;i++){const d=M.cd(s[i],s[n-1])+M.cd(s[i-1],s[n-2])+M.cd(s[i-2],s[n-3]);if(d<best.dist)best={dist:d,next:s[i+1]};}return[best.next];},
  FreqDecay:      s=>{
    const freq={};
    s.forEach((v,i)=>{
      // Exponential recency: more recent = much stronger weight
      freq[v]=(freq[v]||0)+Math.pow(1.6,i);
    });
    // Also add extra weight for last 3 values
    s.slice(-3).forEach(v=>{freq[v]=(freq[v]||0)+5;});
    return Object.entries(freq).sort((a,b)=>b[1]-a[1]).slice(0,2).map(([v])=>parseInt(v));
  },
  Markov:         s=>{
    if(s.length<2)return[s[0]||0];
    const tr={};
    const trDec={};
    const n=s.length;
    for(let i=1;i<n;i++){
      const k=s[i-1];
      // Exponential recency: last 4 = weight 4, next 4 = weight 2, rest = 1
      const w=i>=n-4?4:i>=n-8?2:1;
      if(!tr[k])tr[k]={};
      tr[k][s[i]]=(tr[k][s[i]]||0)+w;
      const dk=Math.floor(s[i-1]/10);
      if(!trDec[dk])trDec[dk]={};
      trDec[dk][Math.floor(s[i]/10)]=(trDec[dk][Math.floor(s[i]/10)]||0)+w;
    }
    const k=s[n-1];
    if(tr[k]&&Object.keys(tr[k]).length>0)
      return Object.entries(tr[k]).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([v])=>parseInt(v));
    const dk=Math.floor(k/10);
    if(trDec[dk]&&Object.keys(trDec[dk]).length>0){
      const bestDecade=parseInt(Object.entries(trDec[dk]).sort((a,b)=>b[1]-a[1])[0][0]);
      const decVals=s.filter(v=>Math.floor(v/10)===bestDecade);
      return[decVals.length?M.mod(Math.round(M.mean(decVals))):bestDecade*10+5];
    }
    return[k];
  },
  Bigram:         s=>{if(s.length<3)return[s[s.length-1]];const tr={};for(let i=1;i<s.length-1;i++){const k=s[i-1]+"_"+s[i];if(!tr[k])tr[k]={};tr[k][s[i+1]]=(tr[k][s[i+1]]||0)+1;}const k=s[s.length-2]+"_"+s[s.length-1];if(!tr[k])return[s[s.length-1]];return Object.entries(tr[k]).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([v])=>parseInt(v));},
  Trigram:        s=>{if(s.length<4)return[s[s.length-1]];const tr={};for(let i=2;i<s.length-1;i++){const k=s[i-2]+"_"+s[i-1]+"_"+s[i];if(!tr[k])tr[k]={};tr[k][s[i+1]]=(tr[k][s[i+1]]||0)+1;}const k=s[s.length-3]+"_"+s[s.length-2]+"_"+s[s.length-1];if(!tr[k])return[s[s.length-1]];return Object.entries(tr[k]).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([v])=>parseInt(v));},
  ZigZag:         s=>{if(s.length<4)return[s[s.length-1]];const n=s.length;let zz=0;
    // Fix: count zigzags up to the last complete triplet (i<n-1 is correct)
    for(let i=1;i<n-1;i++){const a=s[i]-s[i-1],b=s[i+1]-s[i];if((a>0&&b<0)||(a<0&&b>0))zz++;}
    if(zz/(n-2)>0.55){
      // Last gap tells us current direction; reverse it for prediction
      let d=s[n-1]-s[n-2];if(d>50)d-=100;if(d<-50)d+=100;
      // Also consider the gap before to estimate reversal magnitude
      let d2=s[n-2]-s[n-3];if(d2>50)d2-=100;if(d2<-50)d2+=100;
      const mag=Math.round((Math.abs(d)+Math.abs(d2))/2);
      return[M.mod(s[n-1]+(-Math.sign(d))*mag)];
    }
    return[s[n-1]];},
  Sticky:         s=>{const freq={};s.forEach(v=>{freq[v]=(freq[v]||0)+1;});const top=Object.entries(freq).filter(([,c])=>c>=2).sort((a,b)=>b[1]-a[1]).slice(0,4).map(([v])=>parseInt(v));return top.length?top:[s[s.length-1]];},
  XorHeur:        s=>{if(s.length<2)return[s[0]||0];const l=s[s.length-1],p=s[s.length-2];return[(M.d1(l)^M.d1(p))*10+(M.d2(l)^M.d2(p))];},
  RevLag2:        s=>s.length>=3?[M.rev(s[s.length-3])]:[s[s.length-1]],

  // ── PSEUDO-RANDOM STRUCTURE ALGOS (v13 enhanced) ──
  // All now FITTED to data - best parameters found via brute-force search

  // Xorshift: find best shift params via backtest
  Xorshift:       s=>{
    if(s.length<3)return[M.mod(s[s.length-1]^(s[s.length-1]<<3))];
    const n=s.length;let best={sc:-1,a:3,b:5,c:2};
    // Fix: &0x7F (0-127 range) gives better distribution than &0xFF when taking %100
    for(const a of[1,3,5,7,13])for(const b of[3,5,7,9,11])for(const c of[1,2,3,4]){
      let sc=0;
      for(let i=1;i<n;i++){let x=s[i-1]||1;x^=(x<<a)&0x7F;x^=(x>>b)&0x7F;x^=(x<<c)&0x7F;if(M.mod(Math.abs(x))===s[i])sc++;}
      if(sc>best.sc)best={sc,a,b,c};
    }
    let x=s[n-1]||1;x^=(x<<best.a)&0x7F;x^=(x>>best.b)&0x7F;x^=(x<<best.c)&0x7F;
    return[M.mod(Math.abs(x))];
  },

  // MiddleSquare: fit best multiplier
  MiddleSquare:   s=>{
    if(s.length<3)return[parseInt(String(s[s.length-1]*s[s.length-1]).padStart(4,"0").slice(1,3))];
    const n=s.length;let best={sc:-1,m:1};
    for(const m of[1,2,3,5,7,11,13]){
      let sc=0;for(let i=1;i<n;i++){const sq=String((s[i-1]*m)*(s[i-1]*m)).padStart(4,"0");if(parseInt(sq.slice(1,3))===s[i])sc++;}
      if(sc>best.sc)best={sc,m};
    }
    const sq=String((s[n-1]*best.m)*(s[n-1]*best.m)).padStart(4,"0");
    return[M.mod(parseInt(sq.slice(1,3)))];
  },

  // LFSR: try multiple tap positions
  LFSR7:          s=>{
    if(s.length<3)return[M.mod(((s[s.length-1]<<1)|((s[s.length-1]>>6)^(s[s.length-1]>>5))&1)&0x7F)];
    const n=s.length;let best={sc:-1,t1:6,t2:5};
    for(const t1 of[6,5,4])for(const t2 of[5,4,3,2]){
      if(t1<=t2)continue;
      let sc=0;for(let i=1;i<n;i++){const v=s[i-1],bit=((v>>t1)^(v>>t2))&1;if(M.mod(((v<<1)|bit)&0x7F)===s[i])sc++;}
      if(sc>best.sc)best={sc,t1,t2};
    }
    const v=s[n-1],bit=((v>>best.t1)^(v>>best.t2))&1;
    return[M.mod(((v<<1)|bit)&0x7F)];
  },

  // QuadCong: fit best (a,b,c) coefficients
  QuadCong:       s=>{
    if(s.length<3)return[M.mod(3*s[s.length-1]*s[s.length-1]+7*s[s.length-1]+11)];
    const n=s.length;let best={sc:-1,a:3,b:7,c:11};
    for(const a of[1,2,3,5,7])for(const b of[1,3,5,7,11])for(const c of[1,3,7,11,13,17]){
      let sc=0;for(let i=1;i<n;i++)if(M.mod(a*s[i-1]*s[i-1]+b*s[i-1]+c)===s[i])sc++;
      if(sc>best.sc)best={sc,a,b,c};
    }
    return[M.mod(best.a*s[n-1]*s[n-1]+best.b*s[n-1]+best.c)];
  },

  // ParkMiller: fit best multiplier and modulus
  ParkMiller:     s=>{
    if(s.length<3)return[M.mod((16807*s[s.length-1])%97)];
    const n=s.length;let best={sc:-1,a:16807,m:97};
    for(const a of[16807,48271,69621])for(const m of[97,89,83,79,73,67]){
      let sc=0;for(let i=1;i<n;i++)if(M.mod((a*s[i-1])%m)===s[i])sc++;
      if(sc>best.sc)best={sc,a,m};
    }
    return[M.mod((best.a*s[n-1])%best.m)];
  },

  // LagFib: fit best lag offsets (j,k) 
  LagFib:         s=>{
    if(s.length<8)return[s[s.length-1]];
    const n=s.length;let best={sc:-1,j:7,k:3,op:"xor"};
    for(const j of[7,5,4,3])for(const k of[3,2,1])for(const op of["xor","add","sub"]){
      if(j<=k||j>=n)continue;
      let sc=0;
      for(let i=j;i<n;i++){
        let v;
        if(op==="xor")v=s[i-j]^s[i-k];
        else if(op==="add")v=M.mod(s[i-j]+s[i-k]);
        else v=M.mod(s[i-j]-s[i-k]);
        if(v===s[i])sc++;
      }
      if(sc>best.sc)best={sc,j,k,op};
    }
    const sj=s[n-best.j]||0,sk=s[n-best.k]||0;
    const v=best.op==="xor"?sj^sk:best.op==="add"?M.mod(sj+sk):M.mod(sj-sk);
    return[M.mod(v)];
  },

  // Rule30: try rules 30,90,110,150 and pick best fitting
  Rule30:         s=>{
    if(s.length<3)return[s[s.length-1]];
    // Fix: convert rule arrays to Sets for O(1) lookup in inner loop
    const rules={30:new Set([4,3,2,1]),90:new Set([6,3]),110:new Set([6,5,3,2,1]),150:new Set([6,5,4,1])};
    const n=s.length;let best={sc:-1,ruleName:"30",activeRules:[4,3,2,1]};
    for(const[rname,active]of Object.entries(rules)){
      let sc=0;
      for(let i=1;i<n;i++){const v=s[i-1];let out=0;for(let b=0;b<7;b++){const l=(v>>(b+1))&1,c=(v>>b)&1,r=b>0?(v>>(b-1))&1:0,rule=(l<<2)|(c<<1)|r;if(active.has(rule))out|=(1<<b);}if(M.mod(out)===s[i])sc++;}
      if(sc>best.sc)best={sc,ruleName:rname,activeRules:active};
    }
    const v=s[n-1];let out=0;
    for(let b=0;b<7;b++){const l=(v>>(b+1))&1,c=(v>>b)&1,r=b>0?(v>>(b-1))&1:0,rule=(l<<2)|(c<<1)|r;if(best.activeRules.has(rule))out|=(1<<b);}
    return[M.mod(out)];
  },

  // WichmannHill: fit best multipliers
  WichmannHill:   s=>{
    const n=s.length,a=s[n-1]||1,b=n>=2?s[n-2]:1,c=n>=3?s[n-3]:1;
    if(n<4)return[M.mod(Math.floor(((171*a%30269+172*b%30307+170*c%30323)/3%1)*100))];
    let best={sc:-1,ma:171,mb:172,mc:170};
    for(const ma of[171,172,170])for(const mb of[172,171,170])for(const mc of[170,171,172]){
      let sc=0;
      for(let i=3;i<n;i++){
        const ai=s[i-3]||1,bi=s[i-2]||1,ci=s[i-1]||1;
        const x=(ma*ai)%30269,y=(mb*bi)%30307,z=(mc*ci)%30323;
        if(M.mod(Math.floor(((x/30269+y/30307+z/30323)%1)*100))===s[i])sc++;
      }
      if(sc>best.sc)best={sc,ma,mb,mc};
    }
    const x=(best.ma*a)%30269,y=(best.mb*b)%30307,z=(best.mc*c)%30323;
    return[M.mod(Math.floor(((x/30269+y/30307+z/30323)%1)*100))];
  },

  // BBS: fit best semi-prime modulus
  BBS:            s=>{
    if(s.length<3)return[M.mod((s[s.length-1]*s[s.length-1])%87)];
    const n=s.length;
    // Semi-primes (p×q, both ≡ 3 mod 4)
    const mods=[87,91,77,143,209,323];
    let best={sc:-1,mod:87};
    for(const mod of mods){
      let sc=0;for(let i=1;i<n;i++)if(M.mod((s[i-1]*s[i-1])%mod)===s[i])sc++;
      if(sc>best.sc)best={sc,mod};
    }
    return[M.mod((s[n-1]*s[n-1])%best.mod)];
  },

  // MersenneMod: fit best twist parameters
  MersenneMod:    s=>{
    if(s.length<3)return[s[s.length-1]];
    const n=s.length;let best={sc:-1,m1:0x40,m2:0x3F,xv:0x39,mask:0x1F};
    for(const m1 of[0x40,0x20,0x60])for(const xv of[0x39,0x2C,0x15,0x4A]){
      let sc=0;
      for(let i=2;i<n;i++){const v=s[i-1],p=s[i-2];const y=(v&m1)|((p)&(~m1&0x7F));const pred=M.mod((y>>1)^(y&1?xv:0));if(pred===s[i])sc++;}
      if(sc>best.sc)best={sc,m1,xv};
    }
    const v=s[n-1],p=s[n-2];const y=(v&best.m1)|((p)&(~best.m1&0x7F));
    return[M.mod((y>>1)^(y&1?best.xv:0))];
  },

  // ── NEW PRNG FAMILIES (v14 additions) ────────────

  // ICG (Inverse Congruential Generator)
  ICG:            s=>{
    if(s.length<3)return[s[s.length-1]];
    function modInv(x,p){if(x===0)return 0;let r=1,b=p-2,base=x%p;while(b>0){if(b&1)r=r*base%p;base=base*base%p;b>>=1;}return r;}
    const n=s.length;let best={sc:-1,a:1,c:0,p:97};
    const primes=[97,89,83,79,73,71,67,61,59,53,47,43,41];
    outerICG:for(const p of primes){
      // Pre-compute all inverses for this prime once
      const inv=new Array(n);for(let i=0;i<n;i++)inv[i]=modInv(s[i],p);
      for(const a of[1,2,3,5,7,11,13,17,19,23])
        for(const c of[0,1,3,7,11,13,17,23,29,31]){
          let sc=0;
          for(let i=1;i<n;i++){
            if(M.mod(a*inv[i-1]+c)===s[i])sc++;
            else if(sc+(n-i)<best.sc)break;
          }
          if(sc>best.sc){best={sc,a,c,p};if(sc>=n-1)break outerICG;}
        }
    }
    const inv=modInv(s[n-1],best.p);
    return[M.mod(best.a*inv+best.c)];
  },

  // TruncLCG: state=(a*state+c)%M_big, output=floor(state/shift) mod 100
  TruncLCG:       s=>{
    if(s.length<3)return[s[s.length-1]];
    const n=s.length;let best={sc:-1,a:3,c:11,M:967,shift:10};
    const perfect=n-1;
    outerTL:for(const a of[3,5,7,11,13,17,19,23,29,31])
      for(const c of[1,3,7,11,13,17,23,29])
      for(const Mb of[997,991,983,977,971,967,953,947,941,937,929,919,911])
      for(const shift of[1,2,3,5,7,10,13]){
        let sc=0,state=s[0];
        for(let i=1;i<n;i++){
          state=(a*state+c)%Mb;
          if(M.mod(Math.floor(state/shift))===s[i])sc++;
          else if(sc+(n-i)<best.sc)break;
        }
        if(sc>best.sc){best={sc,a,c,M:Mb,shift};if(sc>=perfect)break outerTL;}
      }
    let state=s[n-1];state=(best.a*state+best.c)%best.M;
    return[M.mod(Math.floor(state/best.shift))];
  },

  // SWB (Subtract-With-Borrow): x[n]=x[n-s]-x[n-r]-borrow mod 100
  // A carry-propagating lagged generator — different from LagFib (which uses XOR/add).
  // The borrow bit creates long cycle lengths not found in simple subtraction.
  SWB:            s=>{
    if(s.length<8)return[s[s.length-1]];
    const n=s.length;let best={sc:-1,r:5,ss:3};
    for(const r of[7,6,5,4,3])for(const ss of[3,2,1]){
      if(r<=ss||r>=n)continue;
      let sc=0,borrow=0;
      for(let i=r;i<n;i++){
        const diff=s[i-ss]-s[i-r]-borrow;
        if(M.mod(diff)===s[i])sc++;
        borrow=diff<0?1:0;
      }
      if(sc>best.sc)best={sc,r,ss};
    }
    let borrow=0;
    for(let i=best.r;i<n;i++){const diff=s[i-best.ss]-s[i-best.r]-borrow;borrow=diff<0?1:0;}
    const diff=s[n-best.ss]-s[n-best.r]-borrow;
    return[M.mod(diff)];
  },

  // PCGLike (Permuted Congruential Generator)
  PCGLike:        s=>{
    if(s.length<3)return[s[s.length-1]];
    const n=s.length;let best={sc:-1,a:5,c:11,M:256,k:5};
    const perfect=n-1;
    outerPCG:for(const a of[3,5,7,11,13,17,21,29,37])
      for(const c of[1,3,5,7,11,13,17,23])
      for(const Mb of[128,256,512,1024,2048])
      for(const k of[2,3,4,5,6,7,8,9]){
        let sc=0,state=s[0];
        for(let i=1;i<n;i++){
          state=(a*state+c)%Mb;
          if(M.mod(state^(state>>k))===s[i])sc++;
          else if(sc+(n-i)<best.sc)break;
        }
        if(sc>best.sc){best={sc,a,c,M:Mb,k};if(sc>=perfect)break outerPCG;}
      }
    let state=s[n-1];state=(best.a*state+best.c)%best.M;
    return[M.mod(state^(state>>best.k))];
  },

  // CubicCong (Cubic Congruential): x[n+1]=(a*x^3+b*x+c) mod m
  // Polynomial non-linear congruential — not captured by QuadCong (degree-2).
  // Targets generators using cubic maps over a prime field.
  CubicCong:      s=>{
    if(s.length<3)return[s[s.length-1]];
    const n=s.length;let best={sc:-1,a:1,b:1,c:0,m:97};
    for(const a of[1,2,3,5])for(const b of[1,3,5,7,11,13])for(const c of[0,1,3,7,11,13,17])for(const m of[97,89,83,79,73,67]){
      let sc=0;
      for(let i=1;i<n;i++){const x=s[i-1];if(M.mod((a*x*x*x+b*x+c)%m)===s[i])sc++;}
      if(sc>best.sc)best={sc,a,b,c,m};
    }
    const x=s[n-1];
    return[M.mod((best.a*x*x*x+best.b*x+best.c)%best.m)];
  },

  // RowSeedLCG: x[row]=(a*row + b*row² + c) mod 100
  RowSeedLCG:     s=>{
    if(s.length<4)return[s[s.length-1]];
    const n=s.length;let best={sc:-1,a:1,b:0,c:0};
    const perfect=n*1.4;
    outerRS:for(const a of[1,2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97])
      for(const b of[0,1,2,3,5,7,11,13])
      for(const c of[0,1,3,7,11,13,17,23,29,37,43,53,61,71,79,89,97]){
        let sc=0;
        for(let i=0;i<n;i++){
          const row=i+1;
          const p=M.mod(a*row+b*row*row+c);
          if(p===s[i])sc+=1;
          else if(M.near(p,s[i],2))sc+=0.4;
          else if(sc+(n-i)*1.4<best.sc)break;
        }
        if(sc>best.sc){best={sc,a,b,c};if(sc>=perfect)break outerRS;}
      }
    const nextRow=n+1;
    return[M.mod(best.a*nextRow+best.b*nextRow*nextRow+best.c)];
  },

  // ── PATTERN MEMORY BANK ─────────────────────────
  PatternMemBank: s=>{
    if(s.length<6)return[s[s.length-1]];
    const n=s.length;
    let bestResult=null,bestQuality=-1;
    for(const W of[4,3,2]){
      if(n<=W+1)continue;
      const ctx=s.slice(n-W);
      const matches=[];
      for(let i=W;i<n-1;i++){
        let d=0;for(let j=0;j<W;j++)d+=M.cd(s[i-W+1+j],ctx[j]);
        if(d<=W*6)matches.push({dist:d,next:s[i+1],recency:i});
      }
      if(!matches.length){
        let bst={dist:Infinity,next:s[n-1]};
        for(let i=W;i<n-1;i++){let d=0;for(let j=0;j<W;j++)d+=M.cd(s[i-W+1+j],ctx[j]);if(d<bst.dist)bst={dist:d,next:s[i+1]};}
        if(bst.dist<Infinity){const q=W/(bst.dist+1);if(q>bestQuality){bestQuality=q;bestResult=[bst.next];}}
        continue;
      }
      const quality=matches.reduce((s2,m)=>s2+(1/(m.dist+1)),0);
      if(quality>bestQuality){
        bestQuality=quality;
        const votes={};
        matches.forEach(m=>{const w=(1/(m.dist+1))*(1+m.recency/n);votes[m.next]=(votes[m.next]||0)+w;});
        bestResult=[parseInt(Object.entries(votes).sort((a,b)=>b[1]-a[1])[0][0])];
      }
    }
    return bestResult||[s[n-1]];
  },

  // ── RECURRENCE DISCOVERY ────────────────────────
  ModSearch:      s=>{if(s.length<4)return[s[s.length-1]];const n=s.length;let best={sc:-1,k:2,off:0};for(let k=2;k<=15;k++)for(let off=0;off<100;off+=2){let sc=0;for(let i=1;i<n;i++)if(((s[i-1]%k)+off)%100===s[i])sc++;if(sc>best.sc)best={sc,k,off};}return[M.mod((s[n-1]%best.k)+best.off)];},
  XorChain:       s=>{if(s.length<5)return[s[s.length-1]];const n=s.length;return[M.mod(s[n-1]^s[n-3]^s[n-5])];},
  PolyCong:       s=>{if(s.length<3)return[s[s.length-1]];const n=s.length;let best={sc:-1,a:1,b:1,c:0};for(const a of[1,2,3])for(const b of[1,3,5,7])for(const c of[0,1,3,7,11]){let sc=0;for(let i=1;i<n;i++)if(M.mod(a*s[i-1]*s[i-1]+b*s[i-1]+c)===s[i])sc++;if(sc>best.sc)best={sc,a,b,c};}return[M.mod(best.a*s[n-1]*s[n-1]+best.b*s[n-1]+best.c)];},

  // ── V9 NEW ALGORITHMS ──────────────────────────
  DeepMarkov4:    s=>{
    if(s.length<5)return[s[s.length-1]];
    const n=s.length;
    // Try 4-gram first, fall back to 3-gram, then 2-gram
    for(const depth of[4,3,2]){
      if(n<depth+1)continue;
      const tr={};
      for(let i=depth-1;i<n-1;i++){
        const k=s.slice(i-depth+1,i+1).join("_");
        if(!tr[k])tr[k]={};
        // Recency weight: recent transitions count more (like Markov does)
        const w=i>n-4?2.5:i>n-8?1.5:1;
        tr[k][s[i+1]]=(tr[k][s[i+1]]||0)+w;
      }
      const k=s.slice(n-depth).join("_");
      if(tr[k]&&Object.keys(tr[k]).length>0)
        return Object.entries(tr[k]).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([v])=>parseInt(v));
    }
    return[s[n-1]];
  },
  GapMarkov:      s=>{
    if(s.length<4)return[s[s.length-1]];
    const gaps=[];
    for(let i=1;i<s.length;i++){let g=s[i]-s[i-1];if(g>50)g-=100;if(g<-50)g+=100;gaps.push(g);}
    // Try 2-gram first, fallback to 1-gram
    for(const depth of[2,1]){
      if(gaps.length<depth+1)continue;
      const tr={};
      for(let i=depth-1;i<gaps.length-1;i++){
        const k=gaps.slice(i-depth+1,i+1).join("_");
        if(!tr[k])tr[k]={};
        tr[k][gaps[i+1]]=(tr[k][gaps[i+1]]||0)+1;
      }
      const k=gaps.slice(-depth).join("_");
      if(tr[k]&&Object.keys(tr[k]).length>0){
        const bestGap=parseInt(Object.entries(tr[k]).sort((a,b)=>b[1]-a[1])[0][0]);
        return[M.mod(s[s.length-1]+bestGap)];
      }
    }
    return[s[s.length-1]];
  },
  EntropyAdapt:   s=>{
    if(s.length<6)return[s[s.length-1]];
    const w=s.slice(-8);
    const freq={};w.forEach(v=>{freq[v]=(freq[v]||0)+1;});
    const probs=Object.values(freq).map(c=>c/w.length);
    const entropy=-probs.reduce((sum,p)=>sum+p*Math.log2(p+1e-10),0);
    if(entropy>3.2){
      // high entropy: use frequency
      const f={};s.forEach((v,i)=>{f[v]=(f[v]||0)+Math.pow(1.4,i);});
      return[parseInt(Object.entries(f).sort((a,b)=>b[1]-a[1])[0][0])];
    }
    // low entropy: use linear fit
    const n=s.length;let sx=0,sy=0,sxy=0,sx2=0;
    s.forEach((v,i)=>{sx+=i;sy+=v;sxy+=i*v;sx2+=i*i;});
    const D=n*sx2-sx*sx;if(!D)return[s[n-1]];
    const a=(n*sxy-sx*sy)/D,b=(sy-a*sx)/n;
    return[M.mod(Math.round(a*n+b))];
  },
  FreqMomentum:   s=>{
    if(s.length<8)return[s[s.length-1]];
    const half=Math.floor(s.length/2);
    const old=s.slice(0,half),rec=s.slice(half);
    const fOld={},fRec={};
    old.forEach(v=>{fOld[v]=(fOld[v]||0)+1;});
    rec.forEach(v=>{fRec[v]=(fRec[v]||0)+1;});
    const rising={};
    Object.keys(fRec).forEach(v=>{
      const growth=(fRec[v]||0)-(fOld[v]||0);
      if(growth>0)rising[v]=growth;
    });
    if(!Object.keys(rising).length)return[s[s.length-1]];
    return[parseInt(Object.entries(rising).sort((a,b)=>b[1]-a[1])[0][0])];
  },
  SequenceHash:   s=>{
    if(s.length<5)return[s[s.length-1]];
    const key=s.slice(-4).join(",");
    const hist={};
    for(let i=3;i<s.length-1;i++){
      const k=s.slice(i-3,i+1).join(",");
      if(!hist[k])hist[k]={};
      hist[k][s[i+1]]=(hist[k][s[i+1]]||0)+1;
    }
    if(!hist[key])return[s[s.length-1]];
    return Object.entries(hist[key]).sort((a,b)=>b[1]-a[1]).slice(0,2).map(([v])=>parseInt(v));
  },
  DiffSeriesLin:  s=>{
    if(s.length<4)return[s[s.length-1]];
    const diffs=[];
    for(let i=1;i<s.length;i++){let d=s[i]-s[i-1];if(d>50)d-=100;if(d<-50)d+=100;diffs.push(d);}
    const n=diffs.length;let sx=0,sy=0,sxy=0,sx2=0;
    diffs.forEach((v,i)=>{sx+=i;sy+=v;sxy+=i*v;sx2+=i*i;});
    const D=n*sx2-sx*sx;
    const nextDiff=D?Math.round((n*sxy-sx*sy)/D*n+(sy-(n*sxy-sx*sy)/D*sx)/n):diffs[n-1];
    return[M.mod(s[s.length-1]+nextDiff)];
  },
  ValueCluster:   s=>{
    if(s.length<6)return[s[s.length-1]];
    // k=4 cluster centers: 12,37,62,87
    const centers=[12,37,62,87];
    const last=s[s.length-1];
    const ci=centers.reduce((bi,c,i)=>M.cd(last,c)<M.cd(last,centers[bi])?i:bi,0);
    // find which cluster follows ci most often
    const trans={};
    for(let i=0;i<s.length-1;i++){
      const fc=centers.reduce((bi,c,ix)=>M.cd(s[i],c)<M.cd(s[i],centers[bi])?ix:bi,0);
      const tc=centers.reduce((bi,c,ix)=>M.cd(s[i+1],c)<M.cd(s[i+1],centers[bi])?ix:bi,0);
      trans[fc]=trans[fc]||{};
      trans[fc][tc]=(trans[fc][tc]||0)+1;
    }
    if(!trans[ci])return[centers[ci]];
    const nextCi=parseInt(Object.entries(trans[ci]).sort((a,b)=>b[1]-a[1])[0][0]);
    return[centers[nextCi]];
  },
  // ── V14 REVERSE & SEQUENTIAL PATTERN ALGOS ────────

  // Mirror: reflect value around midpoint 50 (50→50, 30→70, 20→80)
  MirrorAt50:     s=>[M.mod(100-s[s.length-1])],

  // StepDown1: last value minus 1 (detects -1 per row patterns)
  StepDown1:      s=>[M.mod(s[s.length-1]-1)],

  // StepUp1: last value plus 1 (detects +1 per row patterns)
  StepUp1:        s=>[M.mod(s[s.length-1]+1)],

  // StepDown2: minus 2 per step
  StepDown2:      s=>[M.mod(s[s.length-1]-2)],

  // StepUp2: plus 2 per step
  StepUp2:        s=>[M.mod(s[s.length-1]+2)],

  // BestStep: brute-force best constant step (−10 to +10) from history
  BestStep:       s=>{
    if(s.length<3)return[s[s.length-1]];
    let best={sc:-1,step:0};
    for(let step=-15;step<=15;step++){
      let sc=0;
      for(let i=1;i<s.length;i++){
        if(M.mod(s[i-1]+step)===s[i])sc+=1;
        else if(M.near(M.mod(s[i-1]+step),s[i],1))sc+=0.4;
      }
      if(sc>best.sc)best={sc,step};
    }
    return[M.mod(s[s.length-1]+best.step)];
  },

  // AlternatingStep: detects +k, -k, +k, -k pattern (zigzag with fixed amplitude)
  AlternatingStep:s=>{
    if(s.length<4)return[s[s.length-1]];
    const n=s.length;let best={sc:-1,k:1};
    for(let k=1;k<=20;k++){
      let sc=0;
      for(let i=2;i<n;i++){
        const expected=i%2===0?M.mod(s[i-2]):M.mod(s[i-1]+(s[i-1]>s[i-2]?-k:k));
        if(expected===s[i])sc++;
      }
      if(sc>best.sc)best={sc,k};
    }
    // Predict: if last gap was +k, next is -k and vice versa
    const lastGap=s[n-1]-s[n-2];
    return[M.mod(s[n-1]+(lastGap>=0?-best.k:best.k))];
  },

  // SymmetricBounce: detects values bouncing between two walls (lo, hi)
  SymmetricBounce:s=>{
    if(s.length<6)return[s[s.length-1]];
    const lo=Math.min(...s.slice(-8)),hi=Math.max(...s.slice(-8));
    const range=hi-lo;if(range<2)return[s[s.length-1]];
    const v=s[s.length-1];
    // If near top wall, predict going down; near bottom, going up
    const distFromTop=hi-v,distFromBot=v-lo;
    const prevGap=s[s.length-1]-s[s.length-2];
    if(distFromTop<range*0.2&&prevGap>=0)return[M.mod(v-(prevGap||1))]; // reverse at top
    if(distFromBot<range*0.2&&prevGap<=0)return[M.mod(v-(prevGap||1))]; // reverse at bottom
    return[M.mod(v+prevGap)]; // continue trend
  },

  // DecreasingBothSides: detects v, v-1, v-2... or v, v+1, v+2... (arithmetic sequence)
  ArithSeqDetect: s=>{
    if(s.length<4)return[s[s.length-1]];
    const n=s.length;
    // Check last 4 values for consistent step
    const gaps=[];
    for(let i=Math.max(1,n-5);i<n;i++){
      let g=s[i]-s[i-1];if(g>50)g-=100;if(g<-50)g+=100;
      gaps.push(g);
    }
    const firstGap=gaps[0];
    const isConsistent=gaps.every(g=>g===firstGap);
    if(isConsistent&&firstGap!==0)return[M.mod(s[n-1]+firstGap)];
    // Nearly consistent: allow 1 deviation
    const avgGap=Math.round(gaps.reduce((a,b)=>a+b,0)/gaps.length);
    const consistent=gaps.filter(g=>g===avgGap).length;
    if(consistent>=gaps.length-1&&avgGap!==0)return[M.mod(s[n-1]+avgGap)];
    return[s[n-1]];
  },

  // ReverseSequence: detects if sequence is running backwards (99,98,97... or 10,9,8...)
  ReverseSeq:     s=>{
    if(s.length<3)return[s[s.length-1]];
    const n=s.length;
    let decCount=0,incCount=0;
    for(let i=1;i<n;i++){
      let g=s[i]-s[i-1];if(g>50)g-=100;if(g<-50)g+=100;
      if(g===-1)decCount++;
      if(g===1)incCount++;
    }
    if(decCount>=Math.floor(n*0.6))return[M.mod(s[n-1]-1)]; // strong decrease by 1
    if(incCount>=Math.floor(n*0.6))return[M.mod(s[n-1]+1)]; // strong increase by 1
    return[s[n-1]];
  },

  // PalindromeStep: detects values that count up then down (1,2,3,4,3,2,1,2,3...)
  PalindromeStep: s=>{
    if(s.length<5)return[s[s.length-1]];
    const n=s.length;
    const gaps=[];
    for(let i=1;i<n;i++){let g=s[i]-s[i-1];if(g>50)g-=100;if(g<-50)g+=100;gaps.push(g);}
    // Detect direction reversal: if last few gaps are all same sign and previous were opposite
    const last=gaps.slice(-3);
    const prev=gaps.slice(-6,-3);
    if(last.length<2||prev.length<2)return[s[n-1]];
    const lastSign=last.every(g=>g>0)?1:last.every(g=>g<0)?-1:0;
    const prevSign=prev.every(g=>g>0)?1:prev.every(g=>g<0)?-1:0;
    if(lastSign!==0&&prevSign!==0&&lastSign!==prevSign){
      // Currently reversing, continue current direction
      return[M.mod(s[n-1]+lastSign*Math.abs(last[last.length-1]||1))];
    }
    return[s[n-1]];
  },

  // StepAccelerate: detects accelerating step (1,2,4,8 or 1,3,6,10...)
  StepAccelerate: s=>{
    if(s.length<4)return[s[s.length-1]];
    const n=s.length;
    const gaps=[];
    for(let i=1;i<n;i++){let g=s[i]-s[i-1];if(g>50)g-=100;if(g<-50)g+=100;gaps.push(g);}
    const gapGaps=[];
    for(let i=1;i<gaps.length;i++)gapGaps.push(gaps[i]-gaps[i-1]);
    if(!gapGaps.length)return[s[n-1]];
    const avgAccel=Math.round(gapGaps.reduce((a,b)=>a+b,0)/gapGaps.length);
    const nextGap=(gaps[gaps.length-1]||0)+avgAccel;
    // Sanity cap: don't accelerate beyond ±20
    const cappedGap=Math.max(-20,Math.min(20,nextGap));
    return[M.mod(s[n-1]+cappedGap)];
  },

  // DoubleAlternate: detects aa,bb,cc (two same then two same: 11,11,22,22,33,33)
  DoubleAlternate:s=>{
    if(s.length<6)return[s[s.length-1]];
    const n=s.length;
    // Check for pair pattern
    let pairCount=0;
    for(let i=1;i<n-1;i+=2)if(s[i]===s[i-1])pairCount++;
    if(pairCount>=Math.floor((n-1)/2*0.6)){
      // Currently in a pair?
      if(s[n-1]===s[n-2])return[M.mod(s[n-1]+1)]; // pair done, step to next
      return[s[n-1]]; // repeat current value to complete pair
    }
    return[s[n-1]];
  },

  // TripleRepeat: detects aaa,bbb,ccc patterns
  TripleRepeat:   s=>{
    if(s.length<6)return[s[s.length-1]];
    const n=s.length;
    // Check if last 3 are same and before-3 are different
    if(s[n-1]===s[n-2]&&s[n-2]===s[n-3]&&s[n-3]!==s[n-4]){
      // Triple complete, predict increment
      return[M.mod(s[n-1]+1)];
    }
    if(s[n-1]===s[n-2]&&s[n-2]!==s[n-3]){
      // Mid-triple, repeat
      return[s[n-1]];
    }
    return[s[n-1]];
  },

  // SymmetricMirror: detects (a,b) pairs where b=100-a (complements)
  ComplementPairs:s=>{
    if(s.length<4)return[M.mod(100-s[s.length-1])];
    const n=s.length;
    let compCount=0;
    for(let i=1;i<n;i++)if(s[i]+s[i-1]===100)compCount++;
    // If strong complement pattern, predict complement of last value
    if(compCount>=Math.floor((n-1)*0.5))return[M.mod(100-s[n-1])];
    return[s[n-1]];
  },

  // BestModStep: finds best (v mod k + offset) pattern
  ModStep:        s=>{
    if(s.length<4)return[s[s.length-1]];
    const n=s.length;let best={sc:-1,k:2,step:1,off:0};
    for(let k=2;k<=10;k++)for(let step=-k;step<=k;step++)for(let off=0;off<k;off++){
      let sc=0;
      for(let i=1;i<n;i++){
        const expected=M.mod(((s[i-1]%k)+step+k)%k+off);
        if(expected===s[i])sc++;
      }
      if(sc>best.sc)best={sc,k,step,off};
    }
    return[M.mod(((s[n-1]%best.k)+best.step+best.k)%best.k+best.off)];
  },

  // ── V10 NEW ALGORITHMS ──────────────────────────
  NgramVoting:    s=>{
    if(s.length<3)return[s[s.length-1]];
    const votes={};
    // 1-gram
    const f1={};s.forEach(v=>{f1[v]=(f1[v]||0)+1;});
    Object.entries(f1).forEach(([v,c])=>{votes[v]=(votes[v]||0)+c*0.5;});
    // 2-gram
    if(s.length>=2){const k2=s[s.length-2]+"_"+s[s.length-1];const t2={};for(let i=1;i<s.length-1;i++){const k=s[i-1]+"_"+s[i];if(!t2[k])t2[k]={};t2[k][s[i+1]]=(t2[k][s[i+1]]||0)+1;}if(t2[k2])Object.entries(t2[k2]).forEach(([v,c])=>{votes[v]=(votes[v]||0)+c*1.5;});}
    // 3-gram
    if(s.length>=3){const k3=s[s.length-3]+"_"+s[s.length-2]+"_"+s[s.length-1];const t3={};for(let i=2;i<s.length-1;i++){const k=s[i-2]+"_"+s[i-1]+"_"+s[i];if(!t3[k])t3[k]={};t3[k][s[i+1]]=(t3[k][s[i+1]]||0)+1;}if(t3[k3])Object.entries(t3[k3]).forEach(([v,c])=>{votes[v]=(votes[v]||0)+c*3.0;});}
    if(!Object.keys(votes).length)return[s[s.length-1]];
    return Object.entries(votes).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([v])=>parseInt(v));
  },
  EpisodicMemory: s=>{
    if(s.length<10)return[s[s.length-1]];
    const ep=7;const query=s.slice(-ep);
    let best={dist:Infinity,next:s[s.length-1]};
    for(let i=ep;i<s.length-1;i++){
      let d=0;for(let j=0;j<ep;j++)d+=M.cd(s[i-ep+j],query[j]);
      if(d<best.dist)best={dist:d,next:s[i+1]};
    }
    return[best.next];
  },
  TentMap:        s=>{
    if(s.length<3)return[s[s.length-1]];
    const x=s[s.length-1]/99;
    let bestP=0.5,bestSc=-1;
    for(let p=0.3;p<=0.9;p+=0.05){
      let sc=0,xr=s[0]/99;
      for(let i=1;i<s.length;i++){xr=xr<p?xr/p:(1-xr)/(1-p);if(Math.abs(xr*99-s[i])<6)sc++;}
      if(sc>bestSc){bestSc=sc;bestP=p;}
    }
    const nx=x<bestP?x/bestP:(1-x)/(1-bestP);
    return[M.mod(Math.round(nx*99))];
  },

  // ── NEW PRNG DETECTORS ─────────────────────────

  // Combined LCG: x = (a1*x+c1)%m1 XOR (a2*x+c2)%m2 — common in older software
  CombinedLCG:    s=>{
    if(s.length<5)return[s[s.length-1]];
    const n=s.length;
    let best={sc:-1,a1:3,c1:1,m1:89,a2:7,c2:3,m2:97};
    for(const a1 of[3,5,7,11])for(const c1 of[1,3,7,13])for(const m1 of[89,83,79])
    for(const a2 of[7,11,13,17])for(const c2 of[3,7,11,17])for(const m2 of[97,89,83]){
      let sc=0;
      for(let i=1;i<n;i++){
        const pred=M.mod(((a1*s[i-1]+c1)%m1)^((a2*s[i-1]+c2)%m2));
        if(pred===s[i])sc++;
      }
      if(sc>best.sc)best={sc,a1,c1,m1,a2,c2,m2};
    }
    return[M.mod(((best.a1*s[n-1]+best.c1)%best.m1)^((best.a2*s[n-1]+best.c2)%best.m2))];
  },

  // ALFG (Additive Lagged Fibonacci with carry): x[n] = x[n-j] + x[n-k] + carry mod m
  ALFG:           s=>{
    if(s.length<8)return[s[s.length-1]];
    const n=s.length;
    let best={sc:-1,j:7,k:3,m:97};
    for(const j of[7,5,4,3])for(const k of[3,2,1])for(const m of[97,89,83]){
      if(j<=k||j>=n)continue;
      let sc=0,carry=0;
      for(let i=j;i<n;i++){
        const sum=s[i-j]+s[i-k]+carry;
        const pred=M.mod(sum%m);
        carry=Math.floor(sum/m)%2;
        if(pred===s[i])sc++;
      }
      if(sc>best.sc)best={sc,j,k,m};
    }
    let carry=0;
    const sj=s[n-best.j]||0,sk=s[n-best.k]||0;
    const sum=sj+sk+carry;
    return[M.mod(sum%best.m)];
  },

  // DFT Period Detector: finds dominant frequency via simplified DFT
  DFTPeriod:      s=>{
    if(s.length<8)return[s[s.length-1]];
    const n=s.length;
    const avg=M.mean(s);
    let bestPeriod=2,bestPower=0;
    // Test periods 2..floor(n/2), find strongest spectral peak
    for(let p=2;p<=Math.min(16,Math.floor(n/2));p++){
      let re=0,im=0;
      s.forEach((v,i)=>{
        const angle=2*Math.PI*i/p;
        re+=(v-avg)*Math.cos(angle);
        im+=(v-avg)*Math.sin(angle);
      });
      const power=re*re+im*im;
      if(power>bestPower){bestPower=power;bestPeriod=p;}
    }
    // Use detected period for prediction (weighted average of phase-matching positions)
    const p=bestPeriod;
    const positions=[];
    for(let i=n%p||p;i<=n;i+=p)if(s[n-i]!=null)positions.push({v:s[n-i],age:Math.floor(i/p)});
    if(!positions.length)return[s[n-1]];
    let wSum=0,wVal=0;
    positions.forEach(({v,age})=>{const w=Math.exp(-age*0.4);wSum+=w;wVal+=v*w;});
    return[M.mod(Math.round(wVal/wSum))];
  },

  // CrossLag: detects if col values at lag-k predict current col (leading indicator across time)
  // Applied per-series here: s[i-k] vs s[i] for best k
  CrossLagSelf:   s=>{
    if(s.length<6)return[s[s.length-1]];
    const n=s.length;
    // Find lag k (1..5) where s[i-k] best predicts s[i]
    let bestK=1,bestR=0;
    for(let k=1;k<=Math.min(5,Math.floor(n/3));k++){
      const avg1=M.mean(s.slice(0,n-k)),avg2=M.mean(s.slice(k));
      let num=0,d1=0,d2=0;
      for(let i=k;i<n;i++){num+=(s[i-k]-avg1)*(s[i]-avg2);d1+=(s[i-k]-avg1)**2;d2+=(s[i]-avg2)**2;}
      const r=(d1*d2)>0?Math.abs(num/Math.sqrt(d1*d2)):0;
      if(r>bestR){bestR=r;bestK=k;}
    }
    if(bestR<0.2||n<=bestK)return[s[n-1]];
    const lagVals=s.slice(-bestK-3,-bestK).filter(ok);
    const currVals=s.slice(-3).filter(ok);
    if(!lagVals.length||!currVals.length)return[s[n-1]];
    const drift=M.mean(currVals)-M.mean(lagVals);
    const lagPred=s[n-bestK];
    if(!ok(lagPred))return[s[n-1]];
    return[M.mod(Math.round(lagPred+drift))];
  },

  // Bimodal detector: series oscillates between two clusters (low and high band)
  BimodalBounce:  s=>{
    if(s.length<6)return[s[s.length-1]];
    const mid=M.median(s);
    const low=s.filter(v=>v<mid),high=s.filter(v=>v>=mid);
    if(low.length<2||high.length<2)return[s[s.length-1]];
    const loMean=M.mean(low),hiMean=M.mean(high);
    // Empirical: across all months every col splits lo≈25, hi≈75 with ~45-55 gap
    // Count recent band transitions (last 8 values)
    const recent=s.slice(-8);
    let switches=0;
    for(let i=1;i<recent.length;i++){if((recent[i]<mid)!==(recent[i-1]<mid))switches++;}
    const switchRate=switches/(recent.length-1);
    const lastLow=s[s.length-1]<mid;
    if(switchRate>0.45){
      // Alternating — predict opposite cluster
      // Return both the mean AND the most frequent value in target cluster
      const targetCluster=lastLow?high:low;
      const tMean=M.mod(Math.round(M.mean(targetCluster)));
      const freq={};targetCluster.forEach(v=>{freq[v]=(freq[v]||0)+1;});
      const tMode=parseInt(Object.entries(freq).sort((a,b)=>b[1]-a[1])[0][0]);
      return tMean===tMode?[tMean]:[tMean,tMode];
    }
    if(switchRate<0.2&&s.length>=8){
      // Sticky in current band — continue, predict within same cluster
      const sameCluster=lastLow?low:high;
      return[M.mod(Math.round(M.mean(sameCluster)))];
    }
    // Mixed regime: predict whichever band has higher recent frequency
    const recentLow=recent.filter(v=>v<mid).length;
    const recentHigh=recent.filter(v=>v>=mid).length;
    if(recentLow>recentHigh)return[M.mod(Math.round(loMean))];
    return[M.mod(Math.round(hiMean))];
  },

    SumConstraint:  s=>{
    if(s.length<5)return[s[s.length-1]];
    const avg=M.mean(s);
    const std=M.std(s);
    const lo=Math.max(0,Math.round(avg-std));
    const hi=Math.min(99,Math.round(avg+std));
    const last=s[s.length-1];
    if(last<lo)return[M.mod(lo+Math.round((hi-lo)*0.25))];
    if(last>hi)return[M.mod(hi-Math.round((hi-lo)*0.25))];
    let g=last-s[s.length-2];if(g>50)g-=100;if(g<-50)g+=100;
    return[M.mod(last+Math.round(g*0.5))];
  },

  // ── BOUNCE DETECTOR ───────────────────────────
  // Finds the historical ceiling and floor of the series.
  // When near ceiling → predicts downward. Near floor → upward.
  // Critical for bounded 0-99 sequences that oscillate between extremes.
  BounceDetect:   s=>{
    if(s.length<5)return[s[s.length-1]];
    const sorted=[...s].sort((a,b)=>a-b);
    const n=sorted.length;
    // Adaptive percentiles: quartiles for small series, 10/90 for larger
    const loP=n>=12?0.10:0.25;
    const hiP=n>=12?0.90:0.75;
    const floor=sorted[Math.floor(n*loP)];
    const ceil=sorted[Math.floor(n*hiP)];
    const range=ceil-floor||1;
    const last=s[s.length-1];
    const pos=(last-floor)/range; // 0=at floor, 1=at ceil
    if(pos>=0.80){
      // Near ceiling — predict a drop toward midpoint
      const drop=Math.round((last-floor)*0.38); // ~38% retracement
      return[M.mod(last-drop),M.mod(last-Math.round(drop*0.6))];
    }
    if(pos<=0.20){
      // Near floor — predict a rise toward midpoint
      const rise=Math.round((ceil-last)*0.38);
      return[M.mod(last+rise),M.mod(last+Math.round(rise*0.6))];
    }
    // Mid-range: continue recent direction but damped
    let gap=last-s[s.length-2];if(gap>50)gap-=100;if(gap<-50)gap+=100;
    return[M.mod(last+Math.round(gap*0.5))];
  },

  // ── VALUE TRANSITION MATRIX ───────────────────
  // Groups values into decades (00-09, 10-19, ... 90-99).
  // Builds a 10×10 transition table of which decade follows which.
  // Generalizes far better than exact-value Markov with only 31 rows.
  ValTransMatrix: s=>{
    if(s.length<4)return[s[s.length-1]];
    const dec=v=>Math.min(9,Math.floor(v/10));
    const n=s.length;
    // Build weighted transition counts with Laplace smoothing (prevents zero-probability decades)
    const LAPLACE=0.3; // add 0.3 to each cell to smooth sparse data
    const mat=Array.from({length:10},()=>new Array(10).fill(LAPLACE));
    for(let i=1;i<n;i++){
      const age=n-i;
      const w=Math.exp(-age*0.25);
      mat[dec(s[i-1])][dec(s[i])]+=w;
    }
    const fromDec=dec(s[n-1]);
    const row=mat[fromDec];
    const total=row.reduce((a,b)=>a+b,0);
    if(!total)return[s[n-1]];
    // Detect if all cells are just Laplace prior (no real transitions from this decade)
    const realTransitions=s.filter((_,i)=>i>0&&dec(s[i-1])===fromDec).length;
    if(realTransitions===0){
      // No real data for this decade — fall back to global series mean
      return[M.mod(Math.round(M.mean(s)))];
    }
    // Find top 2 destination decades by weight
    const ranked=row.map((w,d)=>({d,w})).sort((a,b)=>b.w-a.w);
    const preds=[];
    for(const {d,w} of ranked.slice(0,2)){
      if(w/total<0.05)break;
      // Predict the historical mean within that target decade
      const decVals=s.filter(v=>dec(v)===d);
      const target=decVals.length?Math.round(M.mean(decVals)):d*10+5;
      preds.push(M.mod(target));
    }
    return preds.length?preds:[s[n-1]];
  },

  // ── KNN WINDOW ────────────────────────────────
  // K-Nearest Neighbors on sliding windows of length 4.
  // Finds the 3 most similar past windows and votes on what came next.
  // Better than EpisodicMemory: uses weighted voting not just best match.
  KNNWindow:      s=>{
    if(s.length<6)return[s[s.length-1]];
    // Adaptive window: use smaller window for short series, larger for longer
    const W=s.length>=20?5:s.length>=12?4:3;
    const n=s.length;
    if(n<=W+1)return[s[n-1]];
    const query=s.slice(-W);
    const matches=[];
    for(let i=W;i<n-1;i++){
      const win=s.slice(i-W,i);
      let dist=0;
      for(let j=0;j<W;j++)dist+=M.cd(win[j],query[j]);
      matches.push({dist,next:s[i],recency:i}); // store raw index as recency
    }
    // Sort by distance, tiebreak by recency (higher index = more recent = better)
    matches.sort((a,b)=>a.dist!==b.dist?a.dist-b.dist:b.recency-a.recency);
    const K=Math.min(5,matches.length);
    const top=matches.slice(0,K);
    const votes={};
    top.forEach((m,rank)=>{
      const w=(1/(m.dist+1))*(1+m.recency/n)/(rank+1);
      votes[m.next]=(votes[m.next]||0)+w;
    });
    return Object.entries(votes)
      .sort((a,b)=>b[1]-a[1])
      .slice(0,2)
      .map(([v])=>parseInt(v));
  },
  // ── BimodalBandPredict: uses fitted lo/hi cluster to predict next band ──
  // Empirical: every col in every month has lo≈25 and hi≈75 bands (std<22 each).
  BimodalBandPredict: s=>{
    if(s.length<8)return[s[s.length-1]];
    const sorted=[...s].sort((a,b)=>a-b);
    const mid=sorted[Math.floor(sorted.length/2)];
    const lo=s.filter(v=>v<mid),hi=s.filter(v=>v>=mid);
    if(lo.length<3||hi.length<3)return[s[s.length-1]];
    const loMean=M.mean(lo),hiMean=M.mean(hi);
    if(hiMean-loMean<20)return[s[s.length-1]];
    const isLo=v=>v<mid;
    let runLen=1;
    for(let i=s.length-2;i>=0&&isLo(s[i])===isLo(s[s.length-1]);i--)runLen++;
    // Estimate average run length from history
    let runTotal=0,runCount=0,cur=1;
    for(let i=1;i<s.length;i++){if(isLo(s[i])===isLo(s[i-1]))cur++;else{runTotal+=cur;runCount++;cur=1;}}
    if(cur>0){runTotal+=cur;runCount++;}
    const avgRun=runCount>1?runTotal/runCount:2;
    const currentLo=isLo(s[s.length-1]);
    if(runLen>=Math.max(1,Math.round(avgRun))){
      const targetCluster=currentLo?hi:lo;
      const freq={};targetCluster.forEach(v=>{freq[v]=(freq[v]||0)+1;});
      const mode=parseInt(Object.entries(freq).sort((a,b)=>b[1]-a[1])[0][0]);
      const tmean=M.mod(Math.round(currentLo?hiMean:loMean));
      return tmean===mode?[tmean]:[tmean,mode];
    }
    const curCluster=currentLo?lo:hi;
    return[M.mod(Math.round(M.mean(curCluster)))];
  },

  // ── PairComplementAlgo: standalone complement-pair candidate generator ──
  // Empirical: A+B≈100 in 17.2% of all rows. Any pair near 100 at ~7-8% each.
  PairComplementAlgo: s=>{
    if(s.length<4)return[s[s.length-1]];
    const v=s[s.length-1];
    return[M.mod(100-v),M.mod(99-v),M.mod(101-v)];
  },

  // ── DigSumPairTarget: ds(pair sum) concentrates on 2-3 values per month ──
  // Empirical: Apr ds(A+B) top=[4(6x),9(6x)]; May top=[8(6x),5(5x)].
  DigSumPairTarget: s=>{
    if(s.length<6)return[s[s.length-1]];
    const n=s.length;
    const ds=v=>{let x=Math.abs(v%100);let r=0;while(x>0){r+=x%10;x=Math.floor(x/10);}return r<10?r:Math.floor(r/10)+r%10;};
    const freq={};
    for(let i=1;i<n;i++)freq[ds(s[i-1]+s[i])]=(freq[ds(s[i-1]+s[i])]||0)+1;
    const topDs=parseInt(Object.entries(freq).sort((a,b)=>b[1]-a[1])[0][0]);
    const last=s[n-1],avg=M.mean(s);
    const candidates=[];
    for(let x=0;x<=99;x++){if(ds(last+x)===topDs)candidates.push(x);}
    if(!candidates.length)return[s[n-1]];
    candidates.sort((a,b)=>Math.abs(a-avg)-Math.abs(b-avg));
    return candidates.slice(0,2);
  },

  // ── StickyPeriod: detects values that repeat at a consistent interval ──
  // Empirical basis: Feb C shows 84 at gap=4 twice, 95 at gap=4 twice → period-4 structure.
  // Feb D shows 88 at gap=5. This finds the dominant repeat gap and predicts accordingly.
  StickyPeriod:   s=>{
    if(s.length<6)return[s[s.length-1]];
    const n=s.length;
    // For each value in recent history, collect all gaps between occurrences
    const gapVotes={};
    const recentWindow=s.slice(-Math.min(n,20));
    recentWindow.forEach(v=>{
      const positions=[];
      for(let i=0;i<n;i++)if(M.cd(s[i],v)<=2)positions.push(i);
      if(positions.length<2)return;
      for(let i=1;i<positions.length;i++){
        const g=positions[i]-positions[i-1];
        if(g>=2&&g<=18){ // realistic period range
          const rw=positions[i]>n-8?2.0:1.0; // recent gaps matter more
          gapVotes[g]=(gapVotes[g]||0)+rw;
        }
      }
    });
    if(!Object.keys(gapVotes).length)return[s[n-1]];
    // Top 2 gap candidates
    const topGaps=Object.entries(gapVotes).sort((a,b)=>b[1]-a[1]).slice(0,2).map(([g])=>parseInt(g));
    const preds=[];
    topGaps.forEach(g=>{
      const srcIdx=n-1-g;
      if(srcIdx>=0){
        // What value came AFTER the last occurrence of s[srcIdx]?
        const srcVal=s[srcIdx];
        const prevOcc=[];
        for(let i=0;i<n-1;i++)if(M.cd(s[i],srcVal)<=2)prevOcc.push(i);
        // Vote on what follows the source value
        const nexts={};
        prevOcc.forEach(pos=>{if(s[pos+1]!=null)nexts[s[pos+1]]=(nexts[s[pos+1]]||0)+1;});
        const topNext=Object.entries(nexts).sort((a,b)=>b[1]-a[1])[0];
        if(topNext)preds.push(parseInt(topNext[0]));
        else preds.push(M.mod(srcVal));
      }
    });
    return preds.length?preds:[s[n-1]];
  },

  // ── DecadeSticky: recent values cluster in a decade; predict within that decade ──
  // Empirical basis: Row02 B consistently in 90s (91,90,96); Row14 B in 80s (82,84,89).
  // Identifies the "home decade" of recent values and predicts within it.
  DecadeSticky:   s=>{
    if(s.length<4)return[s[s.length-1]];
    const recent=s.slice(-10);
    // Weight decades by recency
    const decFreq={};
    recent.forEach((v,i)=>{
      const dec=Math.floor(v/10);
      decFreq[dec]=(decFreq[dec]||0)+Math.pow(1.6,i);
    });
    const topDec=parseInt(Object.entries(decFreq).sort((a,b)=>b[1]-a[1])[0][0]);
    // Within that decade, find the historically most frequent values
    const decVals=s.filter(v=>Math.floor(v/10)===topDec);
    if(decVals.length<2)return[topDec*10+5];
    const valFreq={};
    decVals.forEach((v,i)=>{
      const rw=i>=decVals.length-4?2.0:1.0;
      valFreq[v]=(valFreq[v]||0)+rw;
    });
    const topVals=Object.entries(valFreq).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([v])=>parseInt(v));
    return topVals.length?topVals:[M.mod(Math.round(M.mean(decVals)))];
  },
};
const ALGO_NAMES=Object.keys(A);
const ALGO_COUNT_CONST=ALGO_NAMES.length;
ALGO_COUNT=ALGO_COUNT_CONST;
console.log("Algo count:",ALGO_COUNT);

// ── ALGO FAMILY MAP ───────────────────────────────
// Diversity bonus: when multiple INDEPENDENT families agree on a value, boost it.
// This prevents a cluster of similar algos from dominating just by sheer count.
const _FAM={};
const _FAMS={
  stat:["Mean3","Mean5","WtdMean","Median5","HarmMean","GeoMean","MoveStd","ZScore","ExpSmooth","DblExp","KernelSmooth","MedianFilt","LowPass","BandPass","DiffFilt","LinFit","QuadFit","MovReg","TheilSen","DiffSeriesLin","GapMedian","EntropyAdapt","RetraceRebound","SameRowAvg","SameRowTight","SameRowSnug","SameRowMed","SameRowLast","SameRowWtd","SameRowTrend"],
  seq:["Markov","Bigram","Trigram","DeepMarkov4","PatternMemBank","KNNWindow","SequenceHash","PhaseNN","ValTransMatrix","GapMarkov","EpisodicMem","FreqDecay","Sticky","ValueCluster","StickyPeriod","DecadeSticky","BimodalBandPredict","PairComplementAlgo","DigSumPairTarget"],
  momentum:["WtdMomentum","SecondDiff","LastGap","AutoCorr","Cyclic","AR3","LCGFit","Recurrence2","LogMap","XorChain","ModSearch","BestStep","ArithSeqDetect","StepAccelerate","ZigZag","DFTPeriod","ALFG","CrossLagSelf","FreqMomentum"],
  transform:["Reverse","DigitSum","RevSumTf","MirrorDiff","DigitalRoot","Complement","DigitProduct","RevComplement","SumDoubled","DigSumChain","CubeDigit","DigFact","FibMod","SqrtMod","TriNum","DigSumProd","CollatzStep","XorHeur","RevLag2","SymmetricMirror","BimodalBounce","AlternatingStep","DoubleAlternate","TripleRepeat","PalindromeStep","PairComplementAlgo","DigSumPairTarget"],
  prng:["Xorshift","MiddleSquare","LFSR7","QuadCong","PCGLike","CubicCong","RowSeedLCG","ParkMiller","LagFib","Rule30","WichmannHill","BBS","MersenneMod","ICG","TruncLCG","SWB","PolyCong"],
};
Object.entries(_FAMS).forEach(([fam,names])=>names.forEach(n=>{_FAM[n]=fam;}));
function _getFamily(name){
  if(_FAM[name])return _FAM[name];
  if(name.startsWith("Lin_")||name.startsWith("Gap_")||name.startsWith("Cyc_")||name.startsWith("Rec2_")||name.startsWith("ModStep_"))return"momentum";
  if(name.startsWith("DsChain_")||name.startsWith("Cross_"))return"transform";
  if(name.startsWith("Mut_"))return"seq";
  return"other";
}

// ══════════════════════════════════════════════════
// ── DATE-AWARE ENGINE ─────────────────────────────
// Each row can carry a `date` field (YYYY-MM-DD).
// These algorithms extract calendar features and find
// which date properties correlate with which numbers.
// ══════════════════════════════════════════════════

// Parse a YYYY-MM-DD string into {y,m,d,dow,wom,doy,season,lunar}
function parseDate(dateStr){
  if(!dateStr)return null;
  const dt=new Date(dateStr+"T12:00:00");
  if(isNaN(dt))return null;
  const y=dt.getFullYear(),m=dt.getMonth()+1,d=dt.getDate();
  const dow=dt.getDay(); // 0=Sun..6=Sat
  const wom=Math.ceil(d/7); // week-of-month 1-5
  const doy=Math.floor((dt-(new Date(y,0,1)))/86400000)+1;
  // Season: 0=Winter(Dec-Feb),1=Spring(Mar-May),2=Summer(Jun-Aug),3=Fall(Sep-Nov)
  const season=m<=2||m===12?0:m<=5?1:m<=8?2:3;
  // Lunar phase approximation (synodic cycle ~29.53 days from known new moon)
  const knownNewMoon=new Date("2000-01-06T12:00:00");
  const daysSince=Math.floor((dt-knownNewMoon)/86400000);
  const lunarDay=((daysSince%30)+30)%30; // 0-29
  // 0-7=new,8-14=waxing,15-22=full,23-29=waning
  const lunarPhase=lunarDay<8?0:lunarDay<15?1:lunarDay<23?2:3;
  const lunarPct=Math.round(lunarDay/29.53*100); // 0-100%
  // Digit features of the date
  const dateNum=y*10000+m*100+d;
  const yLast2=y%100;
  const yLast1=y%10;
  const dateDs=M.ds(d)+M.ds(m)+M.ds(yLast2); // sum of digit-sums
  const dateDr=M.dr(d+m+yLast2);
  return{y,m,d,dow,wom,doy,season,lunarDay,lunarPhase,lunarPct,yLast2,yLast1,dateDs,dateDr,
    isWeekend:dow===0||dow===6,isMonthStart:d<=7,isMonthEnd:d>=24,
    monthHalf:d<=15?0:1, // 0=first half, 1=second half
    dayParity:d%2, // 0=even,1=odd
  };
}

function nextDateISO(dateStr){
  if(!dateStr)return"";
  const dt=new Date(dateStr+"T12:00:00");
  if(isNaN(dt))return"";
  dt.setDate(dt.getDate()+1);
  return dt.getFullYear()+"-"+String(dt.getMonth()+1).padStart(2,"0")+"-"+String(dt.getDate()).padStart(2,"0");
}

// ── HELPER: get rows that have a date and a value for `col` ──
function datedRows(col,data){
  return data.filter(r=>ok(r[col])&&r.date&&parseDate(r.date));
}

// ── DATE SIGNAL ENGINE ────────────────────────────
// Returns a dict of {signalName:[predictedValue,...]} for a target date
function getDateSignals(col,data,targetDateStr){
  const res={};
  const td=parseDate(targetDateStr);
  if(!td)return res; // no date info → skip
  const dr=datedRows(col,data);
  if(dr.length<4)return res;

  const vals=dr.map(r=>r[col]);
  const dates=dr.map(r=>parseDate(r.date));

  // ── Pre-group all rows by feature once (single pass, not 18 separate filters) ──
  const byDow={},byMonth={},bySeason={},byLunar={},byDayCluster={},byWom={},byParity={},byHalf={},byWeekend={},byYCycle={};
  dr.forEach((r,i)=>{
    const d=dates[i];if(!d)return;
    const v=r[col];
    (byDow[d.dow]=byDow[d.dow]||[]).push(v);
    (byMonth[d.m]=byMonth[d.m]||[]).push(v);
    (bySeason[d.season]=bySeason[d.season]||[]).push(v);
    (byLunar[d.lunarPhase]=byLunar[d.lunarPhase]||[]).push(v);
    const dc=Math.floor((d.d-1)/7);
    (byDayCluster[dc]=byDayCluster[dc]||[]).push(v);
    (byWom[d.wom]=byWom[d.wom]||[]).push(v);
    (byParity[d.dayParity]=byParity[d.dayParity]||[]).push(v);
    (byHalf[d.monthHalf]=byHalf[d.monthHalf]||[]).push(v);
    const we=d.isWeekend?1:0;
    (byWeekend[we]=byWeekend[we]||[]).push(v);
    const yc=d.y%7;
    (byYCycle[yc]=byYCycle[yc]||[]).push(v);
  });

  // ── 1. DAY-OF-WEEK BIAS ────────────────────────
  const dowBucket=byDow[td.dow]||[];
  if(dowBucket.length>=2){
    const dowAvg=M.mod(Math.round(M.mean(dowBucket)));
    res["DOW_Bias"]=[dowAvg];
    // Also check if DOW correlates with being high/low
    const allAvg=M.mean(vals);
    if(M.mean(dowBucket)>allAvg+5)res["DOW_HighDay"]=[M.mod(Math.round(M.mean(dowBucket)))];
    else if(M.mean(dowBucket)<allAvg-5)res["DOW_LowDay"]=[M.mod(Math.round(M.mean(dowBucket)))];
  }

  // ── 2. MONTH PATTERN ──────────────────────────
  // Average value for this month historically
  const moBucket=byMonth[td.m]||[];
  if(moBucket.length>=2){
    res["Month_Avg"]=[M.mod(Math.round(M.mean(moBucket)))];
    // Month digit sum as transform on the value
    const mDs=M.ds(td.m);
    const moBased=M.mod(Math.round(M.mean(moBucket))+mDs);
    res["Month_DsShift"]=[moBased];
  }

  // ── 3. SEASON SIGNAL ──────────────────────────
  const SEASONS=["Winter","Spring","Summer","Fall"];
  const seaBucket=bySeason[td.season]||[];
  if(seaBucket.length>=2){
    res["Season_"+SEASONS[td.season]]=[M.mod(Math.round(M.mean(seaBucket)))];
  }

  // ── 4. LUNAR PHASE SIGNAL ─────────────────────
  const PHASES=["NewMoon","WaxingQ","FullMoon","WaningQ"];
  const lunBucket=byLunar[td.lunarPhase]||[];
  if(lunBucket.length>=2){
    res["Lunar_"+PHASES[td.lunarPhase]]=[M.mod(Math.round(M.mean(lunBucket)))];
  }

  // ── 5. DATE DIGIT SUM TRANSFORM ───────────────
  // day + month + year_last2 → mod 100
  const dateSum=M.mod(td.d+td.m+td.yLast2);
  res["DateSum_Mod"]=[dateSum];
  res["DateSum_Rev"]=[M.rev(dateSum)];
  res["DateSum_Ds"]=[M.mod(M.ds(td.d)+M.ds(td.m)+M.ds(td.yLast2))];

  // ── 6. DATE DIGITAL ROOT ──────────────────────
  const dateDr=td.dateDr;
  // Find historical values with same digital root
  const dateDrBuckets={};
  dr.forEach((r,i)=>{const d=dates[i];if(d)(dateDrBuckets[d.dateDr]=dateDrBuckets[d.dateDr]||[]).push(r[col]);});
  const drBucket=dateDrBuckets[dateDr]||[];
  if(drBucket.length>=2)res["DateDR_Match"]=[M.mod(Math.round(M.mean(drBucket)))];
  // DR * 11 as a number hint
  res["DateDR_x11"]=[M.mod(dateDr*11)];

  // ── 7. DAY-OF-MONTH RANGE CLUSTER ─────────────
  // 1-7, 8-14, 15-21, 22-28, 29-31 → find avg for this range cluster
  const dayCluster=Math.floor((td.d-1)/7);
  const dcBucket=byDayCluster[dayCluster]||[];
  if(dcBucket.length>=2)res["DayRange_Avg"]=[M.mod(Math.round(M.mean(dcBucket)))];

  // ── 8. WEEK-OF-MONTH SIGNAL ───────────────────
  const womBucket=byWom[td.wom]||[];
  if(womBucket.length>=2)res["WOM_Avg"]=[M.mod(Math.round(M.mean(womBucket)))];

  // ── 9. DAY PARITY (even/odd) ──────────────────
  const parBucket=byParity[td.dayParity]||[];
  if(parBucket.length>=3)res["DayParity_Avg"]=[M.mod(Math.round(M.mean(parBucket)))];

  // ── 10. MONTH HALF (1-15 vs 16-31) ───────────
  const halfBucket=byHalf[td.monthHalf]||[];
  if(halfBucket.length>=3)res["MonthHalf_Avg"]=[M.mod(Math.round(M.mean(halfBucket)))];

  // ── 11. WEEKEND vs WEEKDAY ────────────────────
  const wkBucket=byWeekend[td.isWeekend?1:0]||[];
  if(wkBucket.length>=3)res[td.isWeekend?"Weekend_Avg":"Weekday_Avg"]=[M.mod(Math.round(M.mean(wkBucket)))];

  // ── 12. YEAR CYCLE (year mod 7) ───────────────
  // Detects multi-year repeating cycles (e.g., every 7 years same pattern)
  const yCycle=td.y%7;
  const ycBucket=byYCycle[yCycle]||[];
  if(ycBucket.length>=2)res["YearCycle7"]=[M.mod(Math.round(M.mean(ycBucket)))];

  // ── 13. DATE HASH TRANSFORMS ──────────────────
  // Various numeric combinations of day×month, day+year_last etc.
  res["DxM_Mod"]=[M.mod(td.d*td.m)];
  res["DpM_Ds"]=[M.mod(M.ds(td.d*td.m))];
  res["Day_x3_M"]=[M.mod(td.d*3+td.m*7)];
  const monthEndRows=dr.filter((_,i)=>dates[i]&&dates[i].isMonthEnd).map(r=>r[col]);
  res["YearEnd_Flag"]=td.isMonthEnd&&monthEndRows.length>=2?[M.mod(Math.round(M.mean(monthEndRows)))]:[vals[vals.length-1]||0];

  // ── 14. DAY NUMBER DIRECT TRANSFORM ───────────
  // The row number itself (which = day of month) as algorithm input
  const dayVal=td.d;
  res["Day_Rev"]=[M.rev(dayVal)];
  res["Day_Ds"]=[M.mod(dayVal+M.ds(dayVal))];
  res["Day_Comp"]=[M.mod(100-dayVal)];
  res["Day_x3Mod"]=[M.mod(dayVal*3)];
  res["Day_x7Mod"]=[M.mod(dayVal*7)];

  // ── 15. HISTORICAL SAME-DATE LOOKUP ───────────
  // Same day+month in past years (anniversary effect)
  const anniversaryBucket=dr.filter((_,i)=>dates[i].d===td.d&&dates[i].m===td.m&&dates[i].y!==td.y).map(r=>r[col]);
  if(anniversaryBucket.length>=2)res["Anniversary"]=[M.mod(Math.round(M.mean(anniversaryBucket)))];

  // ── 16. CORRELATION: which date feature best predicts this col? ──
  // Score each date feature against historical values, pick best linear fit
  const features={
    d:dates.map(dt=>dt.d),
    m:dates.map(dt=>dt.m),
    dow:dates.map(dt=>dt.dow),
    lunarDay:dates.map(dt=>dt.lunarDay),
    doy:dates.map(dt=>dt.doy),
    dateDs:dates.map(dt=>dt.dateDs),
  };
  let bestFeat=null,bestR=0;
  Object.entries(features).forEach(([fname,fvals])=>{
    const n=fvals.length;
    const fx=M.mean(fvals),fy=M.mean(vals);
    let num=0,dx=0,dy=0;
    for(let i=0;i<n;i++){num+=(fvals[i]-fx)*(vals[i]-fy);dx+=(fvals[i]-fx)**2;dy+=(vals[i]-fy)**2;}
    const r=Math.sqrt(dx*dy)?num/Math.sqrt(dx*dy):0;
    if(Math.abs(r)>Math.abs(bestR)){bestR=r;bestFeat={name:fname,fvals,r};}
  });
  if(bestFeat&&Math.abs(bestR)>0.15){
    // Simple linear extrapolation using the best correlated date feature
    const n=bestFeat.fvals.length;
    const fx=M.mean(bestFeat.fvals),fy=M.mean(vals);
    let num=0,dx=0;
    for(let i=0;i<n;i++){num+=(bestFeat.fvals[i]-fx)*(vals[i]-fy);dx+=(bestFeat.fvals[i]-fx)**2;}
    const slope=dx?num/dx:0;
    const intercept=fy-slope*fx;
    // Fix: use TARGET DATE's actual feature value, not last historical one
    const tdFeatMap={d:td.d,m:td.m,dow:td.dow,lunarDay:td.lunarDay,doy:td.doy,dateDs:td.dateDs};
    const targetFVal=tdFeatMap[bestFeat.name]??fx;
    const linPred=Math.round(slope*targetFVal+intercept);
    res["BestDateFeat_"+bestFeat.name]=[M.mod(linPred)];
  }

  // ── 17. SAME-DOW SEQUENCE (e.g., every Monday) ──
  // Project next value based on the sequence of all same-weekday values
  if(dowBucket.length>=3){
    const dl=dowBucket.length;
    let dg=dowBucket[dl-1]-dowBucket[dl-2];if(dg>50)dg-=100;if(dg<-50)dg+=100;
    res["DOW_Seq"]=[M.mod(dowBucket[dl-1]+Math.round(dg*0.6))];
  }

  // ── 18. MONTH SEQUENCE (same month year-over-year) ──
  if(moBucket.length>=3){
    const ml=moBucket.length;
    let mg=moBucket[ml-1]-moBucket[ml-2];if(mg>50)mg-=100;if(mg<-50)mg+=100;
    res["Month_Seq"]=[M.mod(moBucket[ml-1]+Math.round(mg*0.5))];
  }

  return res;
}

// Weight multiplier for date signals — calibrated to not overpower per-series algos
// NOTE: Anniversary boosted to 2.4 — same day+month cross-year is DOMINANT in this data
// (Row02 B: 91,90,96 std=3.2; Row14 B: 82,84,89 std=3.6)
const DATE_SIGNAL_WEIGHTS={
  "DOW_Bias":1.4,"DOW_HighDay":1.2,"DOW_LowDay":1.2,"DOW_Seq":1.3,
  "Month_Avg":1.8,"Month_DsShift":1.1,"Month_Seq":1.5,
  "Season_Winter":1.2,"Season_Spring":1.2,"Season_Summer":1.2,"Season_Fall":1.2,
  "Lunar_NewMoon":1.0,"Lunar_WaxingQ":1.0,"Lunar_FullMoon":1.1,"Lunar_WaningQ":1.0,
  "DateSum_Mod":1.3,"DateSum_Rev":1.0,"DateSum_Ds":1.0,
  "DateDR_Match":1.4,"DateDR_x11":0.9,
  "DayRange_Avg":1.5,"WOM_Avg":1.4,"DayParity_Avg":1.1,"MonthHalf_Avg":1.2,
  "Weekend_Avg":1.2,"Weekday_Avg":1.2,
  "YearCycle7":1.1,
  "DxM_Mod":0.9,"DpM_Ds":0.8,"Day_x3_M":0.8,
  "Day_Rev":1.1,"Day_Ds":1.0,"Day_Comp":0.8,"Day_x3Mod":0.8,"Day_x7Mod":0.8,
  "Anniversary":2.4, // same day+month from past years — empirically the strongest single signal
  // ── PATTERN BANK signals — cross-period long-term memory ──────────────
  "PB_MDExact":3.5,   // exact month×day match across years — very strong
  "PB_MDTight":5.0,   // tight std (<5) month×day — near-deterministic
  "PB_DayTight":4.2,  // tight day-of-month across all months
  "PB_DaySnug":3.0,   // snug day-of-month (std<10)
  "PB_DayProfile":2.5,// day-of-month average
  "PB_MonthProfile":2.0, // monthly average
  "PB_MonthTight":3.2,   // tight monthly (std<6)
  "PB_YoY":2.8,       // year-over-year projection
  "PB_YoYDay":3.2,    // YoY + day correction (most precise temporal projection)
  "CrossDsSameDay":3.0,  // same day-of-month from a different dataset
};

// ── SAME-ROW HISTORY ──────────────────────────
// With sequential global day numbers, "same row" now means same DATE or same day-of-month.
// allDatasets: full datasets object — enables cross-period (2025→2026) day matching.
function getSameRowHistory(col,data,predRow,allDatasets){
  const res={};
  const lastRow=data.find(r=>r.row===predRow-1);
  let predDayOfMonth=null,predMonth=null,predYear=null;
  if(lastRow&&lastRow.date){
    const pd=parseDate(lastRow.date);
    if(pd){
      const nd=new Date(lastRow.date+"T12:00:00");nd.setDate(nd.getDate()+1);
      predDayOfMonth=nd.getDate();
      predMonth=nd.getMonth()+1;
      predYear=nd.getFullYear();
    }
  }
  const sameRowExact=data.filter(r=>r.row===predRow&&ok(r[col]));
  const sameDayOfMonth=predDayOfMonth!=null
    ?data.filter(r=>{
        if(!ok(r[col]))return false;
        if(!r.date)return false;
        const pd=parseDate(r.date);
        return pd&&pd.d===predDayOfMonth;
      })
    :sameRowExact;

  // ── CROSS-DATASET same-day search ──────────────────────────────────
  // Critical for 2025→2026 continuity: find day-15 of March 2025 when predicting
  // day-15 of March 2026. These are strong priors the current dataset alone can't provide.
  const crossDsVals=[];
  if(predDayOfMonth!=null&&allDatasets){
    Object.values(allDatasets).forEach(ds=>{
      (ds.rows||[]).forEach(r=>{
        if(!ok(r[col])||!r.date)return;
        const pd=parseDate(r.date);
        if(!pd||pd.d!==predDayOfMonth)return;
        // Don't double-count rows already in current dataset
        if(data.some(x=>x.row===r.row&&x.date===r.date))return;
        crossDsVals.push({v:r[col],m:pd.m,y:pd.y});
      });
    });
  }
  // Same-day-AND-month cross-year (anniversary across datasets)
  const crossAnniversary=crossDsVals.filter(x=>predMonth&&x.m===predMonth);
  // Same-day any-month cross-year
  const crossAnyMonth=crossDsVals;

  const combined=[...new Map([...sameRowExact,...sameDayOfMonth].map(r=>[r.row,r])).values()];

  // Merge cross-dataset values into the pool
  const allVals=[...combined.map(r=>r[col]),...crossDsVals.map(x=>x.v)].filter(v=>ok(v));
  const intraVals=combined.map(r=>r[col]).filter(v=>ok(v));
  if(allVals.length<1)return res;

  const avg=M.mean(allVals);
  const std=M.std(allVals);

  if(intraVals.length>=1){
    res["SameRowAvg"]=[M.mod(Math.round(M.mean(intraVals)))];
    if(M.std(intraVals)<5&&intraVals.length>=3)res["SameRowTight"]=[M.mod(Math.round(M.mean(intraVals)))];
    if(M.std(intraVals)>=5&&M.std(intraVals)<8&&intraVals.length>=3)res["SameRowSnug"]=[M.mod(Math.round(M.median(intraVals)))];
    if(M.std(intraVals)<8&&intraVals.length>=3)res["SameRowMed"]=[M.mod(Math.round(M.median(intraVals)))];
    if(intraVals.length>=2)res["SameRowLast"]=[M.mod(intraVals[intraVals.length-1])];
    if(intraVals.length>=3){
      let ws=0,wv=0;intraVals.forEach((v,i)=>{const w=Math.pow(2,i);ws+=w;wv+=v*w;});
      res["SameRowWtd"]=[M.mod(Math.round(wv/ws))];
    }
    if(intraVals.length>=3){
      const n=intraVals.length;let g=intraVals[n-1]-intraVals[n-2];
      if(g>50)g-=100;if(g<-50)g+=100;
      res["SameRowTrend"]=[M.mod(intraVals[n-1]+Math.round(g*0.4))];
    }
  }

  // Cross-dataset anniversary (same month + same day, different year) — very strong signal
  if(crossAnniversary.length>=1){
    const ann=crossAnniversary.map(x=>x.v);
    res["CrossDsAnniversary"]=[M.mod(Math.round(M.mean(ann)))];
    if(M.std(ann)<6&&ann.length>=2)res["CrossDsAnnivTight"]=[M.mod(Math.round(M.mean(ann)))];
  }

  // Cross-dataset same-day (any month) — weaker but still informative
  if(crossAnyMonth.length>=3){
    res["CrossDsSameDay"]=[M.mod(Math.round(M.mean(crossAnyMonth.map(x=>x.v))))];
  }

  return res;
}

// ── ROW SUM TARGET ────────────────────────────
// All 4 cols come from the same hidden source per row.
// If A+B+C+D orbits a stable sum, use it to constrain
// predictions for any unknown column.
// e.g. if sum ≈ 180 and A=40,B=50,C=45, then D ≈ 45
function getRowSumSignal(col,data,knownPreds){
  const res={};
  // Collect rows where all 4 values are known
  const complete=data.filter(r=>COLS.every(c=>ok(r[c])));
  if(complete.length<4)return res;
  const sums=complete.map(r=>r.A+r.B+r.C+r.D);
  const avgSum=M.mean(sums);
  const stdSum=M.std(sums);
  // Only use this signal if sum is reasonably stable
  // Empirical: May has tightest std=39.2; Jan has loosest std=67.5 — relax threshold to 70
  if(stdSum>70)return res;
  // Known values for this prediction round (other cols already predicted)
  const otherCols=COLS.filter(c=>c!==col);
  const knownSum=otherCols.reduce((acc,c)=>{
    const v=knownPreds&&knownPreds[c]!=null?knownPreds[c]:null;
    return v!=null?acc+v:acc;
  },0);
  const knownCount=otherCols.filter(c=>knownPreds&&knownPreds[c]!=null).length;
  if(knownCount<2)return res; // need at least 2 other cols known
  const implied=Math.round(avgSum-knownSum);
  if(implied>=0&&implied<=99){
    res["RowSumTarget"]=[M.mod(implied)];
    // Also try sum±1std
    const lo=Math.round((avgSum-stdSum*0.5)-knownSum);
    const hi=Math.round((avgSum+stdSum*0.5)-knownSum);
    if(lo>=0&&lo<=99&&lo!==implied)res["RowSumLo"]=[M.mod(lo)];
    if(hi>=0&&hi<=99&&hi!==implied)res["RowSumHi"]=[M.mod(hi)];
  }
  return res;
}

// ── COL GAP STABILITY ─────────────────────────
// B−A, C−B, D−C gaps may be near-constant since all 4
// cols are generated by the same hidden algorithm.
// Detects stable pairwise gaps and uses them to predict.
function getColGapSignals(col,data){
  const ck=col+"_"+data.length+"_"+_TC._ver;
  if(_TC.cg[ck])return _TC.cg[ck];
  const res={};
  const last=data.length?data[data.length-1]:null;
  if(!last)return(_TC.cg[ck]=res);
  const srcCols=COLS.filter(c=>c!==col&&ok(last[c]));
  if(!srcCols.length)return(_TC.cg[ck]=res);
  const gapSums={},gapSumSq={},gapCounts={},lastGaps={};
  srcCols.forEach(src=>{gapSums[src]=0;gapSumSq[src]=0;gapCounts[src]=0;lastGaps[src]=null;});
  data.forEach(r=>{
    srcCols.forEach(src=>{
      if(!ok(r[src])||!ok(r[col]))return;
      let g=r[col]-r[src];if(g>50)g-=100;if(g<-50)g+=100;
      gapSums[src]+=g;gapSumSq[src]+=g*g;gapCounts[src]++;lastGaps[src]=g;
    });
  });
  srcCols.forEach(src=>{
    const cnt=gapCounts[src];if(cnt<4)return;
    const gMean=gapSums[src]/cnt;
    const gVar=gapSumSq[src]/cnt-gMean*gMean;
    const gStd=gVar>0?Math.sqrt(gVar*(cnt/(cnt-1))):0;
    if(gStd>12)return;
    res["Gap_"+src+"to"+col]=[M.mod(Math.round(last[src]+gMean))];
    const lg=lastGaps[src];
    if(lg!=null&&Math.abs(lg-gMean)<=gStd)res["LastGap_"+src+"to"+col]=[M.mod(Math.round(last[src]+lg))];
  });
  _TC.cg[ck]=res;
  return res;
}

// ── TEMPORAL CONSTANTS ─────────────────────────
// A=06:00(360min), B=18:00(1080min), C=21:00(1260min), D=23:50(1430min)
// Gap to next row's A (next day 06:00 = 1800min from current A)
// Recency weights for predicting next-A: D is freshest (370min ago), C(540), B(720), A(1440)
const T_MINS={A:360,B:1080,C:1260,D:1430};
const T_TO_NEXT_A={A:1440,B:720,C:540,D:370}; // minutes until next row's A
// Normalized recency weight (smaller gap = higher weight, exponential)
function tWeight(col){return Math.exp(-T_TO_NEXT_A[col]/800);}
// Gap between two columns in same row (minutes)
function tGap(c1,c2){return Math.abs(T_MINS[c2]-T_MINS[c1]);}

// ── INTRA-ROW TRANSFORM MINER ──────────────────
// For a pair (src,tgt), test ~40 transforms and find which fits best historically.
// Returns {name, pred, hitRate}
const TRANSFORMS=[
  {n:"Add",  f:(a,b)=>M.mod(a+b)},
  {n:"Sub",  f:(a,b)=>M.mod(a-b)},
  {n:"SubR", f:(a,b)=>M.mod(b-a)},
  {n:"DsA",  f:(a)=>M.mod(a+M.ds(a))},
  {n:"DsB",  f:(_,b)=>M.mod(b+M.ds(b))},
  {n:"RevA", f:(a)=>M.rev(a)},
  {n:"RevB", f:(_,b)=>M.rev(b)},
  {n:"RevSum",f:(a,b)=>M.mod(M.rev(a)+M.rev(b))},
  {n:"RevDiff",f:(a,b)=>M.mod(Math.abs(M.rev(a)-M.rev(b)))},
  {n:"Comp", f:(a)=>M.mod(100-a)},
  {n:"CompB",f:(_,b)=>M.mod(100-b)},
  {n:"Half", f:(a)=>M.mod(Math.round(a/2))},
  {n:"Dbl",  f:(a)=>M.mod(a*2)},
  {n:"Dr11", f:(a)=>M.mod(M.dr(a)*11)},
  {n:"Dp",   f:(a)=>M.mod(M.dp(a)||a+1)},
  {n:"XorD", f:(a,b)=>M.mod((M.d1(a)^M.d1(b))*10+(M.d2(a)^M.d2(b)))},
  {n:"Mean", f:(a,b)=>M.mod(Math.round((a+b)/2))},
  {n:"SumDs",f:(a,b)=>M.mod(M.ds(a)+M.ds(b))},
  {n:"AbsDf",f:(a,b)=>M.mod(Math.abs(a-b))},
  {n:"DiffDs",f:(a,b)=>M.mod(Math.abs(M.ds(a)-M.ds(b)))},
  {n:"ProdD",f:(a,b)=>M.mod(M.d1(a)*M.d2(b)+M.d2(a)*M.d1(b))},
  {n:"RevMn",f:(a,b)=>M.mod(Math.round((M.rev(a)+b)/2))},
  {n:"SumM3",f:(a,b)=>M.mod((a+b)%37)},
  {n:"SumM7",f:(a,b)=>M.mod((a+b)%73)},
  {n:"SumM9",f:(a,b)=>M.mod((a+b)%89)},
  {n:"DsSum",f:(a,b)=>{let x=M.ds(a+b);while(x>=10)x=M.ds(x);return x*11;}},
  {n:"RotA", f:(a)=>M.mod((a+13)%100)},
  {n:"RotB", f:(_,b)=>M.mod((b+13)%100)},
  {n:"ModA", f:(a)=>M.mod(a%37)},
  {n:"RevAB",f:(a,b)=>M.mod(M.rev(M.mod(a+b)))},
];
function mineTransform(srcCol,tgtCol,data){
  const ck=srcCol+"_"+tgtCol+"_"+data.length+"_"+_TC._ver;
  if(_TC.tx[ck]!==undefined)return _TC.tx[ck];
  const rows=data.filter(r=>ok(r[srcCol])&&ok(r[tgtCol]));
  if(rows.length<4)return(_TC.tx[ck]=null);
  // Cap at last 80 rows — older data adds noise not signal
  const recent=rows.length>80?rows.slice(-80):rows;
  let best={n:"",score:-1,pred:null};
  for(const t of TRANSFORMS){
    let hits=0,nearhits=0;
    recent.forEach(r=>{
      const p=t.f(r[srcCol],r[tgtCol]);
      if(p===r[tgtCol])hits++;
      else if(M.near(p,r[tgtCol],3))nearhits++;
    });
    const score=(hits+(nearhits*0.4))/recent.length;
    if(score>best.score){
      const lr=recent[recent.length-1];
      best={n:t.n+"("+srcCol+"→"+tgtCol+")",score,pred:t.f(lr[srcCol],lr[tgtCol]),hits,total:recent.length};
    }
  }
  const result=best.score>0.12?best:null;
  _TC.tx[ck]=result;
  return result;
}

// ── SHARED ROW PROPERTY DETECTOR ───────────────
// Checks if all 4 cols in a row share a property (same digit root, same mod class, etc.)
// Returns an array of consistent predictor functions for the target col.
function getSharedRowProps(targetCol,data){
  const ck=targetCol+"_"+data.length+"_"+_TC._ver;
  if(_TC.sp[ck])return _TC.sp[ck];
  const res=[];
  const complete=data.filter(r=>COLS.every(c=>ok(r[c])));
  if(complete.length<4)return(_TC.sp[ck]=res);
  let drMatches=0;
  complete.forEach(r=>{const roots=COLS.map(c=>M.dr(r[c]));if(new Set(roots).size===1)drMatches++;});
  if(drMatches/complete.length>0.3)res.push({name:"SharedDR",pred:()=>{
    const last=complete[complete.length-1];
    const knownDr=M.dr(last[COLS.find(c=>c!==targetCol&&ok(last[c]))||"A"]);
    const ser=getSeries(targetCol,data);const avg=Math.round(M.mean(ser));
    for(let delta=0;delta<=50;delta++){for(const v of[M.mod(avg+delta),M.mod(avg-delta)]){if(M.dr(v)===knownDr)return[v];}}
    return null;
  }});
  const adjPairs=[["A","B"],["B","C"],["C","D"]];
  adjPairs.forEach(([c1,c2])=>{
    if(c1===targetCol||c2===targetCol){
      let dsSame=0;
      complete.forEach(r=>{if(M.ds(r[c1])===M.ds(r[c2]))dsSame++;});
      if(dsSame/complete.length>0.35)res.push({name:"SameDs_"+c1+c2,pred:()=>{
        const src=c1===targetCol?c2:c1;const last=complete[complete.length-1];
        if(!ok(last[src]))return null;const targetDs=M.ds(last[src]);
        const ser=getSeries(targetCol,data);const avg=Math.round(M.mean(ser));
        for(let d=0;d<=50;d++){for(const v of[M.mod(avg+d),M.mod(avg-d)]){if(M.ds(v)===targetDs)return[v];}}
        return null;
      }});
    }
  });
  _TC.sp[ck]=res;
  return res;
}

// ── TEMPORAL RECENCY CHAIN ─────────────────────
// Builds a time-weighted signal from ALL other cols of the SAME row toward targetCol.
// D is most recent before next A. Uses gap-decay: closer in time = stronger signal.
function getTemporalChain(targetCol,data){
  const res={};
  const last=data.length?data[data.length-1]:null;
  const prev=data.length>1?data[data.length-2]:null;
  if(!last)return res;
  const otherCols=COLS.filter(c=>c!==targetCol&&ok(last[c]));
  if(!otherCols.length)return res;

  // 1. Gap-decay weighted blend of other cols in last row → predict targetCol
  // Weight = e^(-gapMinutes/600). D→A has gap 370min (highest weight)
  let wSum=0,wVal=0;
  otherCols.forEach(c=>{
    const gap=tGap(c,targetCol)||300;
    const w=Math.exp(-gap/600);
    wSum+=w;wVal+=last[c]*w;
  });
  if(wSum>0)res["TempBlend"]=[M.mod(Math.round(wVal/wSum))];

  // 2. Most recent column (smallest time gap to targetCol) as direct signal
  const nearest=otherCols.sort((a,b)=>tGap(a,targetCol)-tGap(b,targetCol))[0];
  if(nearest){
    res["NearestCol"]=[last[nearest]];
    res["NearestRev"]=[M.rev(last[nearest])];
    res["NearestComp"]=[M.mod(100-last[nearest])];
    res["NearestDs"]=[M.mod(last[nearest]+M.ds(last[nearest]))];
  }

  // 3. D→A recency link: when predicting A, D from previous row is freshest signal (370min)
  if(targetCol==="A"&&prev&&ok(prev.D)){
    const gap=T_TO_NEXT_A["D"]; // 370 min — shortest gap before next A
    res["D_to_nextA"]=[prev.D];
    res["D_to_nextA_Rev"]=[M.rev(prev.D)];
    res["D_to_nextA_Ds"]=[M.mod(prev.D+M.ds(prev.D))];
  }
  // B→A: 720min gap
  if(targetCol==="A"&&prev&&ok(prev.B)){
    res["B_to_nextA"]=[prev.B];
  }
  // C→A: 540min gap — medium freshness
  if(targetCol==="A"&&prev&&ok(prev.C)){
    res["C_to_nextA"]=[M.mod(prev.C+M.ds(prev.C))];
  }

  // 4. Intra-row transform predictions from each source col → targetCol
  for(const srcCol of COLS.filter(c=>c!==targetCol)){
    const hit=mineTransform(srcCol,targetCol,data);
    if(hit&&hit.pred!=null){
      // Weight by temporal closeness AND transform hit rate
      const tScore=Math.exp(-tGap(srcCol,targetCol)/700)*hit.score;
      if(tScore>0.04)res["Tx_"+hit.n]=[M.mod(Math.round(hit.pred))];
    }
  }

  // 5. Shared row property predictions
  const sharedProps=getSharedRowProps(targetCol,data);
  sharedProps.forEach(sp=>{
    try{const p=sp.pred();if(p)res[sp.name]=p;}catch(e){}
  });

  // 6. Cross-row temporal pattern: same column, weighted by recency of each row
  // More recent rows contribute more (exponential decay by row age)
  const ser=getSeries(targetCol,data);
  if(ser.length>=4){
    let wS=0,wV=0;
    ser.forEach((v,i)=>{const age=ser.length-1-i;const w=Math.exp(-age*0.35);wS+=w;wV+=v*w;});
    res["TempRecency"]=[M.mod(Math.round(wV/wS))];
  }

  // 7. Legacy cross-col (keep backward compat)
  const ci=COLS.indexOf(targetCol),aR=COLS[(ci+1)%4],aL=COLS[(ci+3)%4];
  if(ok(last[aL]))res.RevAdj=[M.rev(last[aL])];
  if(ok(last[aR])){res.DsAdj=[M.mod(ser?.[ser.length-1]+M.ds(last[aR]))];res.AdjRevSum=[M.mod(M.rev(last[aR])+(ser?.[ser.length-1]||0))];}
  if(ok(last[aR])&&ok(last[aL])){
    res.SubCols=[M.mod(last[aL]-last[aR]),M.mod(last[aR]-last[aL])];
    res.XorCols=[M.mod((M.d1(last[aL])^M.d1(last[aR]))*10+(M.d2(last[aL])^M.d2(last[aR])))];
  }
  if(prev&&ok(prev[targetCol])&&ok(prev[aR])&&ok(last[aR]))res.LagDelta=[M.mod(last[aR]+(prev[targetCol]-prev[aR]))];
  if(COLS.every(c=>ok(last[c]))){
    res.RowSum=[M.mod(last.A+last.B+last.C+last.D)];
    res.RowHash=[M.mod((last.A*3+last.B*7+last.C*11+last.D*13)%100)];
  }
  [["A","B"],["C","D"],["B","D"],["A","C"],["A","D"],["B","C"]].forEach(([c1,c2])=>{
    if(ok(last[c1])&&ok(last[c2]))res[c1+c2+"Sum"]=[M.mod(last[c1]+last[c2])];
  });

  // best correlated column (keep legacy)
  let bestCorr=null,bestCol=null;
  COLS.filter(c=>c!==targetCol).forEach(c=>{
    const sx=getSeries(targetCol,data),sy=getSeries(c,data),n=Math.min(sx.length,sy.length);
    if(n<4)return;
    const ax=M.mean(sx.slice(-n)),ay=M.mean(sy.slice(-n));
    let num=0,dx=0,dy=0;
    for(let i=0;i<n;i++){num+=(sx[sx.length-n+i]-ax)*(sy[sy.length-n+i]-ay);dx+=(sx[sx.length-n+i]-ax)**2;dy+=(sy[sy.length-n+i]-ay)**2;}
    const corr=Math.sqrt(dx*dy)?num/Math.sqrt(dx*dy):0;
    if(bestCorr==null||Math.abs(corr)>Math.abs(bestCorr)){bestCorr=corr;bestCol=c;}
  });
  if(bestCol&&bestCorr!=null&&Math.abs(bestCorr)>=0.18&&ok(last[bestCol]))res["Corr_"+bestCol]=[M.mod(last[bestCol])];

  // 8. ── CROSS-COL COMPLEMENT PAIR DETECTOR ──────────────────────────────
  // Empirical: A+B≈100 in 17.2% of all rows (strongest pair), CD≈100 in 14.8%.
  // Tests targets 98-102 for each pair to find the most consistent sum.
  {
    const pairs=COLS.filter(c=>c!==targetCol);
    let bestPairCol=null,bestPairHits=0,bestTarget=100;
    const complete=data.filter(r=>COLS.every(c=>ok(r[c])));
    if(complete.length>=5){
      pairs.forEach(partnerCol=>{
        for(const target of[98,99,100,101,102]){
          let hits=0;
          complete.forEach(r=>{if(Math.abs(r[targetCol]+r[partnerCol]-target)<=3)hits++;});
          if(hits>bestPairHits){bestPairHits=hits;bestPairCol=partnerCol;bestTarget=target;}
        }
      });
      const threshold=Math.max(3,Math.ceil(complete.length*0.12));
      if(bestPairCol&&bestPairHits>=threshold&&ok(last[bestPairCol])){
        res["ComplementPair_"+bestPairCol]=[M.mod(bestTarget-last[bestPairCol])];
      }
    }
  }

  return res;
}

// ── CROSS-COL (now delegates to temporal chain) ─
function getCross(col,data){return getTemporalChain(col,data);}

// ── BACKTEST ───────────────────────────────────
function btScore(fn,series){
  // Hard cap: 150 rows max — beyond this accuracy gain is negligible, cost grows O(n)
  const s=series.length>150?series.slice(-150):series;
  const n=s.length;if(n<5)return 0.05;
  // Only backtest last 40 steps (was 31 — slightly wider for better signal)
  const from=Math.max(3,n-40);let score=0,cnt=0;
  const hist=s.slice(0,from);
  const recentStd=n>=8?M.std(s.slice(-8)):15;
  const nearTol=recentStd>18?3:2;
  for(let i=from;i<n;i++){
    try{
      const p=fn(hist),a=s[i];
      const pm=p.length===1?1.0:p.length===2?0.8:p.length===3?0.65:0.5;
      const age=n-1-i;
      const rw=age<4?3.0:age<8?2.0:age<15?1.4:1.0;
      if(p.some(v=>v===a))score+=(1.0*pm)*rw;
      else if(p.some(v=>M.near(v,a,nearTol)))score+=(0.4*pm)*rw;
      if(p[0]===a)score+=0.35*rw;
      cnt+=rw;
      hist.push(s[i]);
    }catch(e){hist.push(s[i]);}
  }
  return cnt?score/cnt:0.05;
}

function walkFwd(fn,series){
  const n=series.length,h=Math.min(5,Math.floor(n*0.25));
  if(n<h+4)return null;
  // Build history once, extend incrementally — avoids [...train,...slice] spread per iter
  const hist=series.slice(0,n-h);
  let ex=0,nr=0,top1ex=0;
  for(let i=0;i<h;i++){
    try{
      const p=fn(hist),a=series[n-h+i];
      if(p.some(v=>v===a))ex++;
      else if(p.some(v=>M.near(v,a,2)))nr++;
      if(p[0]===a)top1ex++;
      hist.push(a);
    }catch(e){hist.push(series[n-h+i]||0);}
  }
  return{exact:ex,near:nr,top1:top1ex,total:h,pct:Math.round((ex+nr*0.4)/h*100),top1pct:Math.round(top1ex/h*100)};
}

function buildCorr(data){
  const m={};
  COLS.forEach(c1=>{m[c1]={};COLS.forEach(c2=>{
    const sx=getSeries(c1,data),sy=getSeries(c2,data),n=Math.min(sx.length,sy.length);
    if(n<4){m[c1][c2]=0;return;}
    const ax=M.mean(sx.slice(-n)),ay=M.mean(sy.slice(-n));
    let num=0,dx=0,dy=0;
    for(let i=0;i<n;i++){num+=(sx[sx.length-n+i]-ax)*(sy[sy.length-n+i]-ay);dx+=(sx[sx.length-n+i]-ax)**2;dy+=(sy[sy.length-n+i]-ay)**2;}
    m[c1][c2]=Math.sqrt(dx*dy)?+(num/Math.sqrt(dx*dy)).toFixed(2):0;
  });});
  return m;
}

// ── CUSTOM FN EVAL ─────────────────────────────
function makeCustomFn(code){
  try{
    const fn=new Function("s","M","try{const r=("+code+")(s,M);return Array.isArray(r)?r.map(v=>M.mod(Math.round(v))):[M.mod(Math.round(r))];}catch(e){return[s[s.length-1]];}");
    return s=>fn(s,M);
  }catch(e){return null;}
}

// ── STACKED META-LEARNER ──────────────────────
function buildMetaModel(accLog){
  if(!accLog||accLog.length<3)return null;
  // Cap to last 80 sessions — older ones hurt EMA accuracy more than they help
  const log=accLog.length>80?accLog.slice(-80):accLog;
  const ema={},count={};
  const alpha=log.length<10?0.30:log.length<30?0.25:0.18;
  log.forEach(entry=>{
    if(!entry.algoDetails)return;
    COLS.forEach(col=>{
      const actual=entry.actuals&&entry.actuals[col];
      if(!ok(actual))return;
      const details=entry.algoDetails[col];
      if(!details)return;
      Object.entries(details).forEach(([name,pred])=>{
        if(!ok(pred))return;
        count[name]=(count[name]||0)+1;
        const ex=M.mod(Math.round(pred))===actual;
        const nr=!ex&&M.near(M.mod(Math.round(pred)),actual,2);
        const hit=ex?1:nr?0.4:0;
        ema[name]=count[name]===1?0.15*(1-alpha)+alpha*hit:(1-alpha)*(ema[name]||0.15)+alpha*hit;
      });
    });
  });
  const weights={};
  Object.keys(count).forEach(name=>{weights[name]=count[name]>=2?ema[name]:null;});
  return{weights,trained:log.length,alpha};
}

// Fix 9: calibration accuracy → learning rate multiplier
function getCalibMult(conf,cal){
  if(!cal||!cal[conf]||cal[conf].total<5)return 1.0;
  const rate=cal[conf].right/cal[conf].total;
  const baseline=conf==="HIGH"?0.40:conf==="MED"?0.25:0.10;
  return Math.max(0.4,Math.min(1.8,0.5+rate/baseline*0.5));
}

// ── JOINT COLUMN PREDICTOR ─────────────────────
function jointColHint(col,data,knownPreds){
  // Find rows in history where other col values are close to current known preds
  if(!data.length)return null;
  const otherCols=COLS.filter(c=>c!==col&&knownPreds[c]!=null);
  if(!otherCols.length)return null;
  const scored=data.filter(r=>ok(r[col])).map(r=>{
    const dist=otherCols.reduce((s,c)=>s+M.cd(r[c]||0,knownPreds[c]),0);
    return{v:r[col],dist};
  }).sort((a,b)=>a.dist-b.dist).slice(0,5);
  if(!scored.length)return null;
  const totalW=scored.reduce((s,x)=>s+1/(x.dist+1),0);
  const wv=scored.reduce((s,x)=>s+x.v/(x.dist+1),0);
  return M.mod(Math.round(wv/totalW));
}

// ── FORGETTING CURVE ───────────────────────────
// Bug 1 fix: was checking prediction VALUES ("42","73") against algo NAMES ("Markov")
// — they never matched so decay never fired. Now collects actual algo names.
function applyForgetting(weights,accLog){
  if(!accLog||accLog.length<3)return weights;
  const next={...weights};
  // Collect which algo NAMES contributed to top predictions in recent sessions
  const recentNames=new Set();
  accLog.slice(-5).forEach(e=>{
    COLS.forEach(col=>{
      // algoDetails stores {algoName: predictedValue} — keys are algo names
      const details=e.algoDetails&&e.algoDetails[col];
      if(!details)return;
      const actual=e.actuals&&e.actuals[col];
      if(!ok(actual))return;
      // Mark algos that were accurate recently
      Object.entries(details).forEach(([name,pred])=>{
        if(ok(pred)&&M.cd(M.mod(Math.round(pred)),actual)<=3)recentNames.add(name);
      });
    });
  });
  // Decay algos that haven't been accurate recently
  Object.keys(next).filter(k=>!k.startsWith('_')).forEach(name=>{
    if(!recentNames.has(name)&&next[name]>1.0){
      next[name]=Math.max(0.5,next[name]*0.93);
    }
  });
  return next;
}

// ── ROW DIFFICULTY ─────────────────────────────
function computeRowDifficulty(accLog){
  const rowStats={};
  (accLog||[]).forEach(e=>{
    const r=e.targetRow;if(!r)return;
    if(!rowStats[r])rowStats[r]={exact:0,total:0};
    rowStats[r].total++;
    rowStats[r].exact+=e.exactCount;
  });
  const difficulty={};
  Object.entries(rowStats).forEach(([r,s])=>{
    if(s.total>=2)difficulty[r]=Math.round(s.exact/(s.total*4)*100);
  });
  return difficulty;
}

// ── REGIME DETECTION ──────────────────────────
function getRegime(series){
  if(series.length<6)return"normal";
  const r=series.slice(-7),o=series.slice(-14,-7);
  const stdR=M.std(r),stdO=M.std(o.length>=2?o:[r[0]]);
  if(stdR>stdO*1.5)return"volatile";
  const gaps=[];
  for(let i=1;i<r.length;i++){let g=r[i]-r[i-1];if(g>50)g-=100;if(g<-50)g+=100;gaps.push(Math.abs(g));}
  const avgGap=M.mean(gaps);
  if(avgGap<3)return"flat";
  if(avgGap>20)return"trending";
  // Bimodal detection: if values cluster strongly in two separate bands
  // Empirical: ALL columns in ALL months show bimodal with gap≈45-55
  if(series.length>=6){
    const med=M.median(series);
    const low=series.filter(v=>v<med),high=series.filter(v=>v>=med);
    if(low.length>=3&&high.length>=3){
      const gapBetween=M.mean(high)-M.mean(low);
      const spreadLow=M.std(low),spreadHigh=M.std(high);
      // Lower thresholds: gap>20 and each cluster std<25 (empirically justified)
      if(gapBetween>20&&spreadLow<25&&spreadHigh<25)return"bimodal";
    }
  }
  return"normal";
}
// Regime-gated algo pool: FULL exclusion not just multipliers
const REGIME_POOLS={
  volatile:new Set(["Mean3","Mean5","WtdMean","Median5","HarmMean","GeoMean","MoveStd","ZScore","ExpSmooth","DblExp","KernelSmooth","MedianFilt","LowPass","BandPass","FreqDecay","Sticky","FreqMomentum","ValueCluster","SumConstraint","EntropyAdapt","DFTPeriod","StickyPeriod","DecadeSticky","BimodalBandPredict","PairComplementAlgo","DigSumPairTarget"]),
  flat:new Set(["Markov","Bigram","Trigram","DeepMarkov4","SequenceHash","PatternMemBank","FreqDecay","Sticky","FreqMomentum","GapMarkov","AutoCorr","LagFib","XorChain","ModSearch","KNNWindow","CrossLagSelf","StickyPeriod"]),
  trending:new Set(["WtdMomentum","SecondDiff","LastGap","GapMedian","TheilSen","LinFit","QuadFit","MovReg","DiffSeriesLin","AR3","DblExp","LCGFit","Recurrence2","Cyclic","LogMap","PhaseNN","DFTPeriod","ALFG"]),
  bimodal:new Set(["BimodalBounce","BimodalBandPredict","DecadeSticky","StickyPeriod","ValTransMatrix","Markov","DeepMarkov4","KNNWindow","PatternMemBank","FreqDecay","Sticky","ValueCluster","PairComplementAlgo","DigSumPairTarget","EntropyAdapt","WtdMean","Mean5","Median5"]),
  normal:null
};
function algoAllowed(name,regime){
  const pool=REGIME_POOLS[regime];
  if(!pool)return true;
  return pool.has(name);
}
function rankAlgoForRegime(name,regime){
  let sc=CORE_ALGO_PRIORITY.includes(name)?2.5:1.0;
  if(regime==="flat"&&/Markov|Pattern|Sticky|KNN|CrossLag|Mode|Recency/i.test(name))sc+=1.4;
  if(regime==="volatile"&&/Mean|Median|Smooth|Entropy|Band|Cluster/i.test(name))sc+=1.4;
  if(regime==="trending"&&/Momentum|Gap|Lin|Quad|Theil|AR|Diff|Cyclic/i.test(name))sc+=1.4;
  if(regime==="bimodal"&&/Bimodal|Decade|Sticky|Cluster|Pattern/i.test(name))sc+=1.4;
  return sc;
}
function selectAlgoNames(names,regime,budget){
  const ranked=[...names].sort((a,b)=>rankAlgoForRegime(b,regime)-rankAlgoForRegime(a,regime));
  const cap=Math.min(Math.max(8,budget||ranked.length),ranked.length);
  return ranked.slice(0,cap);
}

// ── TEMPORAL_BOOST: module-level constant (not recreated per call) ──
const TEMPORAL_BOOST={
  "TempBlend":2.2,"NearestCol":2.0,"NearestRev":1.7,"NearestDs":1.7,"NearestComp":1.5,
  "D_to_nextA":2.5,"D_to_nextA_Rev":1.8,"D_to_nextA_Ds":1.8,"B_to_nextA":1.6,"C_to_nextA":1.7,
  "TempRecency":1.9,"SharedDR":1.6,"SameDs_AB":1.5,"SameDs_BC":1.5,"SameDs_CD":1.5,
  "ComplementPair_A":2.2,"ComplementPair_B":2.2,"ComplementPair_C":2.2,"ComplementPair_D":2.2,
  "BimodalBandPredict":1.8,"BimodalBounce":1.6,
  // ── Cross-dataset anniversary signals (critical for 2025→2026 continuity) ──
  "CrossDsAnniversary":3.8, // same month+day across years in different datasets
  "CrossDsAnnivTight":5.2,  // tight std — near-deterministic anniversary
  "CrossDsSameDay":2.5,
  // ── XL cross-lag ──
  "XL_A_lag0":2.0,"XL_B_lag0":2.0,"XL_C_lag0":2.0,
  "XL_A_lag1":1.8,"XL_B_lag1":1.8,"XL_C_lag1":1.8,
};

// ── PREDICT ────────────────────────────────────
// allDatasets: S.datasets — enables cross-period global series for btScore
// patternBank: S.patternBank — long-term distilled patterns
function predictCol(col,data,W,customs,targetDate,allDatasets,patternBank){
  const tStart=PERF_NOW();
  const series=getSeries(col,data);
  if(series.length<3)return null;

  // ── GLOBAL SERIES: spans ALL datasets for superior algo calibration ──
  // When 2026 data is sparse (10 rows), global series adds 150 rows from 2025.
  // btScore on globalSeries gives far better algo ranking than 10 rows alone.
  const globalSeries=allDatasets?getGlobalSeries(col,allDatasets):series;
  const btSeries=globalSeries.length>series.length?globalSeries:series;

  const gw=W.global||{},rw=W.perRow||{},rnw=W.perRange||{};
  const ns=W.neuralScores||{};
  const maxRow=data.length?data.reduce((m,r)=>r.row>m?r.row:m,0)||1:1;
  const predRow=maxRow+1;
  const curRange=Math.floor((series[series.length-1]||0)/25);
  const regime=getRegime(series);
  const votes={},_contribSets={},details={};
  const cast=(name,val,w)=>{
    const v=M.mod(Math.round(val));
    votes[v]=(votes[v]||0)+w;
    if(!_contribSets[v])_contribSets[v]=new Set();
    _contribSets[v].add(name);
  };
  const rgw=W.perRegime&&W.perRegime[regime]?W.perRegime[regime]:{};
  const algoCache={};
  const perf={btMs:0,builtinMs:0,crossMs:0,dateMs:0,customMs:0,totalMs:0};
  const lightweight=!!W._lightweight;
  const allowedAlgos=ALGO_NAMES.filter(n=>algoAllowed(n,regime));
  const autoBudget=lightweight?12:series.length>240?16:series.length>140?22:series.length>80?30:44;
  const algoBudget=Math.max(8,Math.min(W._algoBudget||autoBudget,allowedAlgos.length));
  const evalNames=selectAlgoNames(allowedAlgos,regime,algoBudget);

  // ── Pre-cache: use globalSeries for btScore to leverage all historical data ──
  if(btSeries.length>=5){
    const btT0=PERF_NOW();
    evalNames.forEach(name=>{
      const fn=A[name];
      try{
        const cacheKey=col+"__"+name;
        if(W._btCache&&W._btCache[cacheKey]){
          algoCache[name]=W._btCache[cacheKey];
          return;
        }
        const bt=btScore(fn,btSeries); // ← uses global series
        let wfBoost=1.0;
        if(!lightweight&&btSeries.length>=10){
          const wf=walkFwd(fn,btSeries);
          if(wf&&wf.total>=3){
            const wfRate=(wf.exact+wf.near*0.4)/wf.total;
            wfBoost=0.6+wfRate*0.8;
          }
        }
        algoCache[name]={bt,wfBoost};
      }catch(e){algoCache[name]={bt:0.05,wfBoost:1.0};}
    });
    perf.btMs+=PERF_NOW()-btT0;
  }

  // built-in algos — predictions run on LOCAL series (recent context), scored on global
  const builtinT0=PERF_NOW();
  evalNames.forEach(name=>{
    const fn=A[name];
    try{
      const cached=algoCache[name]||{bt:0.05,wfBoost:1.0};
      const {bt,wfBoost}=cached;
      const lw=gw[name]!=null?gw[name]:1;
      const rowW=rw[predRow]?rw[predRow][name]!=null?rw[predRow][name]:1:1;
      const ranW=rnw[curRange]?rnw[curRange][name]!=null?rnw[curRange][name]:1:1;
      const regW=rgw[name]!=null?rgw[name]:1;
      const nScore=ns[name]!=null?ns[name]:0;
      const lb=W._leaderboard&&W._leaderboard[name]!=null?W._leaderboard[name]:0;
      const lbMult=lb>2?1.35:lb>1?1.2:lb>0.4?1.08:lb<0.05?0.85:1.0;
      const nMult=nScore>2.5?2.0:nScore>1.5?1.7:nScore>0.5?1.3:nScore>0?1.1:nScore<-1.5?0.35:nScore<-1?0.5:nScore<-0.3?0.75:1.0;
      const btFactor=0.2+Math.sqrt(bt)*3.5;
      const w=btFactor*wfBoost*Math.max(0.05,lw)*Math.max(0.1,rowW)*Math.max(0.1,ranW)*Math.max(0.1,regW)*nMult*lbMult;
      const preds=fn(series);
      preds.forEach((p,i)=>cast(name,p,w/(i*0.6+1)));
      details[name]={pred:preds[0],bt:Math.round(bt*100),lw:+lw.toFixed(2),rw:+rowW.toFixed(2),w:+w.toFixed(2),type:"builtin"};
    }catch(e){}
  });
  perf.builtinMs+=PERF_NOW()-builtinT0;

  // cross-col (temporal-aware)
  const crossT0=PERF_NOW();
  const cross=getCross(col,data);
  Object.entries(cross).forEach(([name,preds])=>{
    if(!preds||!preds.length||!ok(preds[0]))return;
    const lw=gw[name]!=null?gw[name]:1;
    const tBoost=TEMPORAL_BOOST[name]||(name.startsWith("Tx_")?2.0:name.startsWith("Corr_")?1.6:name.startsWith("XL_")?2.2:1.8);
    const signalQ=name.startsWith("Tx_")?0.9:name.startsWith("Corr_")?0.85:name.startsWith("Gap_")?0.8:0.75;
    if(lightweight&&signalQ<0.8)return;
    const w=tBoost*Math.max(0.05,lw)*signalQ;
    if(w<MIN_CROSS_SIGNAL_WEIGHT)return;
    preds.forEach((p,i)=>cast(name,p,w/(i*0.5+1)));
    const isTemp=!!(TEMPORAL_BOOST[name]||name.startsWith("Tx_")||name.startsWith("XL_"));
    details[name]={pred:preds[0],bt:null,lw:+lw.toFixed(2),rw:1,w:+w.toFixed(2),type:isTemp?"temporal":"cross"};
  });
  perf.crossMs+=PERF_NOW()-crossT0;

  // ── DATE SIGNALS ────────────────────────────────
  const dateT0=PERF_NOW();
  if(targetDate){
    const dateSigs=getDateSignals(col,data,targetDate);
    Object.entries(dateSigs).forEach(([name,preds])=>{
      if(!preds||!preds.length||!ok(preds[0]))return;
      const lw=gw[name]!=null?gw[name]:1;
      const baseW=DATE_SIGNAL_WEIGHTS[name]||1.0;
      const w=baseW*Math.max(0.05,lw);
      preds.forEach((p,i)=>cast(name,p,w/(i*0.5+1)));
      details[name]={pred:preds[0],bt:null,lw:+lw.toFixed(2),rw:1,w:+w.toFixed(2),type:"date"};
    });
  }
  perf.dateMs+=PERF_NOW()-dateT0;

  // ── PATTERN BANK SIGNALS — long-term cross-period memory ─────────────
  // These are the strongest cross-year signals: month×day profiles from 2025 data
  // directly inform 2026 predictions even when 2026 has only a few rows.
  if(patternBank){
    const pbSigs=getPatternBankSignals(col,patternBank,targetDate);
    Object.entries(pbSigs).forEach(([name,preds])=>{
      if(!preds||!preds[0]||!ok(preds[0]))return;
      const lw=gw[name]!=null?gw[name]:1;
      const baseW=DATE_SIGNAL_WEIGHTS[name]||2.0;
      const w=baseW*Math.max(0.05,lw);
      preds.forEach((p,i)=>{if(ok(p))cast(name,p,w/(i*0.5+1));});
      details[name]={pred:preds[0],bt:null,lw:+lw.toFixed(2),rw:1,w:+w.toFixed(2),type:"patternbank"};
    });
  }

  // ── DATE-CONDITIONED WEIGHT BOOST ─────────────────────────────────────
  if(targetDate){
    const pd=parseDate(targetDate);
    if(pd){
      const dow=pd.dow,lunar=pd.lunarPhase;
      const perDOW=W.perDOW&&W.perDOW[dow]?W.perDOW[dow]:{};
      const perLunar=W.perLunar&&W.perLunar[lunar]?W.perLunar[lunar]:{};
      Object.keys(details).forEach(name=>{
        const dw=perDOW[name]!=null?perDOW[name]:1;
        const lw2=perLunar[name]!=null?perLunar[name]:1;
        const boost=Math.max(0.1,dw)*Math.max(0.1,lw2);
        if(boost!==1&&details[name]){
          const pred2=details[name].pred;
          if(ok(pred2)){const v=M.mod(Math.round(pred2));if(votes[v])votes[v]*=boost;}
        }
      });
    }
  }

  // ── SAME-ROW HISTORY (cross-dataset aware) ────────────────────────────
  {
    const sameRowSigs=getSameRowHistory(col,data,predRow,allDatasets);
    Object.entries(sameRowSigs).forEach(([name,preds])=>{
      const lw=gw[name]!=null?gw[name]:1;
      // Expanded weights: anniversary signals are very high
      const baseW=name==="CrossDsAnnivTight"?5.5
        :name==="CrossDsAnniversary"?4.0
        :name==="CrossDsSameDay"?2.8
        :name==="SameRowTight"?4.2
        :name==="SameRowSnug"?3.2
        :name==="SameRowAvg"?2.8
        :name==="SameRowWtd"?2.6
        :name==="SameRowLast"?2.4
        :name==="SameRowMed"?2.2
        :1.8; // SameRowTrend
      const w=baseW*Math.max(0.05,lw);
      preds.forEach((p,i)=>cast(name,p,w/(i*0.5+1)));
      details[name]={pred:preds[0],bt:null,lw:+lw.toFixed(2),rw:1,w:+w.toFixed(2),
        type:name.startsWith("CrossDs")?"patternbank":"rowhistory"};
    });
  }

  // ── COL GAP STABILITY ─────────────────────────────────────────────────
  {
    const gapSigs=getColGapSignals(col,data);
    Object.entries(gapSigs).forEach(([name,preds])=>{
      const lw=gw[name]!=null?gw[name]:1;
      const baseW=name.startsWith("Gap_")?2.0:1.6;
      const w=baseW*Math.max(0.05,lw);
      preds.forEach((p,i)=>cast(name,p,w/(i*0.5+1)));
      details[name]={pred:preds[0],bt:null,lw:+lw.toFixed(2),rw:1,w:+w.toFixed(2),type:"colgap"};
    });
  }

  // Pre-cache custom algos (uses globalSeries for btScore)
  const customT0=PERF_NOW();
  const activeCustoms=(customs||[]).filter(ca=>ca.enabled&&ca.code);
  const customBudget=Math.max(6,Math.min(lightweight?10:20,activeCustoms.length));
  const scopedCustoms=activeCustoms
    .map(ca=>({ca,score:(gw[ca.name]||1)+(ns[ca.name]||0)}))
    .sort((a,b)=>b.score-a.score).slice(0,customBudget).map(x=>x.ca);
  scopedCustoms.forEach(ca=>{
    if(algoCache[ca.name])return;
    try{
      const fn=makeCustomFn(ca.code);if(!fn)return;
      const cacheKey=col+"__"+ca.name;
      if(W._btCache&&W._btCache[cacheKey]){algoCache[ca.name]=W._btCache[cacheKey];return;}
      const bt=btScore(fn,btSeries); // ← global series
      let wfBoost=1.0;
      if(!lightweight&&btSeries.length>=10){
        const wf=walkFwd(fn,btSeries);
        if(wf&&wf.total>=3){const wfRate=(wf.exact+wf.near*0.4)/wf.total;wfBoost=0.6+wfRate*0.8;}
      }
      algoCache[ca.name]={bt,wfBoost,fn};
    }catch(e){}
  });
  // custom algos — run predictions on local series
  scopedCustoms.forEach(ca=>{
    if(!ca.enabled||!ca.code)return;
    try{
      const cached=algoCache[ca.name];if(!cached)return;
      const {bt,wfBoost,fn}=cached;
      if(!fn)return;
      const lw=gw[ca.name]!=null?gw[ca.name]:1;
      const rowW=rw[predRow]?rw[predRow][ca.name]!=null?rw[predRow][ca.name]:1:1;
      const ranW=rnw[curRange]?rnw[curRange][ca.name]!=null?rnw[curRange][ca.name]:1:1;
      const regW=rgw[ca.name]!=null?rgw[ca.name]:1;
      const nScore=ns[ca.name]!=null?ns[ca.name]:0;
      const lb=W._leaderboard&&W._leaderboard[ca.name]!=null?W._leaderboard[ca.name]:0;
      const lbMult=lb>2?1.35:lb>1?1.2:lb>0.4?1.08:lb<0.05?0.85:1.0;
      const nMult=nScore>2.5?2.0:nScore>1.5?1.7:nScore>0.5?1.3:nScore>0?1.1:nScore<-1.5?0.35:nScore<-1?0.5:nScore<-0.3?0.75:1.0;
      const btFactor=0.2+Math.sqrt(bt)*3.5;
      const w=btFactor*wfBoost*Math.max(0.05,lw)*Math.max(0.1,rowW)*Math.max(0.1,ranW)*Math.max(0.1,regW)*nMult*lbMult;
      const preds=fn(series);
      preds.forEach((p,i)=>cast(ca.name,p,w/(i*0.6+1)));
      details[ca.name]={pred:preds[0],bt:Math.round(bt*100),lw:+lw.toFixed(2),rw:+rowW.toFixed(2),w:+w.toFixed(2),type:"custom"};
    }catch(e){}
  });
  perf.customMs+=PERF_NOW()-customT0;

  // Adaptive: redistribute tiny votes (< 1% of max) to reduce noise
  const maxVote=Math.max(...Object.values(votes));
  Object.keys(votes).forEach(v=>{if(votes[v]<maxVote*0.01)delete votes[v];});

  // ── FAMILY DIVERSITY BONUS ────────────────────────────────────────────
  const votesByFamily={};
  Object.entries(details).forEach(([name,info])=>{
    const v=info.pred!=null?M.mod(Math.round(info.pred)):-1;
    if(v<0||!votes[v])return;
    const fam=_getFamily(name);
    if(!votesByFamily[v])votesByFamily[v]=new Set();
    votesByFamily[v].add(fam);
  });
  const FAM_MULT=[1,1,1.25,1.55,1.85,2.2];
  Object.keys(votesByFamily).forEach(v=>{
    const vi=parseInt(v);
    const nFam=votesByFamily[v].size;
    if(nFam>=2&&votes[vi])votes[vi]*=FAM_MULT[Math.min(nFam,5)];
  });

  // ── MULTI-LAYER HARMONY AMPLIFIER ─────────────────────────────────────
  // The deepest insight: when INDEPENDENT evidence streams (algo families, date signals,
  // pattern bank, cross-dataset history, cross-col temporal) ALL point to the same value,
  // that convergence is exponentially more reliable than any one stream alone.
  // This is the "all algos working in harmony" mechanism.
  {
    const signalTypeSupport={}; // value → Set of evidence stream types
    Object.entries(details).forEach(([,info])=>{
      if(!ok(info.pred))return;
      const v=M.mod(Math.round(info.pred));
      if(!signalTypeSupport[v])signalTypeSupport[v]=new Set();
      signalTypeSupport[v].add(info.type||"builtin");
    });
    // Multi-layer multipliers — each additional independent stream adds compounding confidence
    // 2 types → 1.5x, 3 → 2.1x, 4 → 2.9x, 5+ → 3.8x
    const LAYER_MULT=[1,1,1.5,2.1,2.9,3.8];
    Object.entries(signalTypeSupport).forEach(([v,types])=>{
      const vi=parseInt(v);
      if(!votes[vi])return;
      const n=types.size;
      if(n>=2)votes[vi]*=LAYER_MULT[Math.min(n,5)];
    });
  }

  // ── CONSENSUS FILTER: weighted agreement ──────────────────────────────
  const algoAgreement={};
  Object.entries(details).forEach(([,info])=>{
    const v=info.pred!=null?M.mod(Math.round(info.pred)):-1;
    if(v>=0)algoAgreement[v]=(algoAgreement[v]||0)+(info.w||1);
  });
  const maxAgree=Math.max(1,...Object.values(algoAgreement));
  Object.entries(algoAgreement).forEach(([v,wt])=>{
    const agreeRatio=wt/maxAgree;
    if(agreeRatio>0.3&&votes[parseInt(v)])votes[parseInt(v)]*=1+agreeRatio*0.4;
  });

  // Meta-learner
  const metaModel=W._metaModel||null;
  if(metaModel){
    Object.keys(votes).forEach(v=>{
      const vInt=parseInt(v);
      const supporters=Object.entries(details).filter(([,d])=>ok(d.pred)&&M.mod(Math.round(d.pred))===vInt);
      supporters.forEach(([name])=>{
        const rate=metaModel.weights[name];
        if(rate==null)return;
        const mult=rate>0.55?1.5:rate>0.40?1.25:rate>0.25?1.05:rate<0.08?0.5:0.8;
        votes[vInt]*=mult;
      });
    });
  }

  // ── HISTORICAL FREQ FILTER ────────────────────────────────────────────
  // Uses globalSeries frequency for more accurate "has this value ever appeared?"
  const freqMap={};
  btSeries.forEach(v=>{freqMap[v]=(freqMap[v]||0)+1;});
  Object.keys(votes).forEach(v=>{
    const vInt=parseInt(v);
    const seenCount=freqMap[vInt]||0;
    if(seenCount===0)votes[vInt]*=0.35;
    else if(seenCount===1)votes[vInt]*=0.7;
  });

  // ── DEAD-ZONE BIAS CORRECTION ─────────────────────────────────────────
  if(W._accLog&&W._accLog.length>=4){
    const recentErrs=W._accLog.slice(-6).map(e=>{
      const pred=e.preds&&e.preds[col];const act=e.actuals&&e.actuals[col];
      if(!ok(pred)||!ok(act))return null;
      let err=act-pred;if(err>50)err-=100;if(err<-50)err+=100;
      return err;
    }).filter(e=>e!=null);
    if(recentErrs.length>=4){
      const posCount=recentErrs.filter(e=>e>2).length;
      const negCount=recentErrs.filter(e=>e<-2).length;
      const n=recentErrs.length;
      if(posCount>=Math.ceil(n*0.66)||negCount>=Math.ceil(n*0.66)){
        const consistencyRatio=Math.max(posCount,negCount)/n;
        const avgErr=Math.round(M.mean(recentErrs)*0.5*consistencyRatio);
        Object.keys(votes).forEach(v=>{
          const shifted=M.mod(parseInt(v)+avgErr);
          if(votes[parseInt(v)]&&shifted!==parseInt(v)){
            votes[shifted]=(votes[shifted]||0)+votes[parseInt(v)]*0.5;
          }
        });
      }
    }
  }

  // ── ENSEMBLE VARIANCE → CONFIDENCE DOWNGRADE ─────────────────────────
  const _allPreds=Object.values(details).map(d=>d.pred).filter(ok);
  const _ensembleVar=_allPreds.length>3?+M.std(_allPreds).toFixed(1):0;

  const total=Object.values(votes).reduce((a,b)=>a+b,0)||1;
  const top5=Object.entries(votes).sort((a,b)=>b[1]-a[1]).slice(0,5)
    .map(([v,vt])=>({value:parseInt(v),votes:+vt.toFixed(2),pct:Math.round(vt/total*100),algos:_contribSets[v]?[..._contribSets[v]]:[]}));

  const ac=Object.keys(details).length;
  const consensus=top5[0]?Math.round(top5[0].algos.length/ac*100):0;
  const t1pct=top5[0]?top5[0].pct:0;
  const highThr=Math.max(12,40-ac*0.18);
  const medThr=Math.max(6,20-ac*0.09);
  const top1FamSupport=top5[0]?((votesByFamily[top5[0].value]||new Set()).size):0;
  const conf=_ensembleVar>25?"LOW"
    :t1pct>highThr&&_ensembleVar<12&&top1FamSupport>=2?"HIGH"
    :t1pct>medThr||top1FamSupport>=3?"MED"
    :"LOW";
  const confClr=conf==="HIGH"?"#34d399":conf==="MED"?"#fbbf24":"#f87171";
  const allP=Object.values(details).map(d=>d.pred).filter(ok);
  const sAllP=[...allP].sort((a,b)=>a-b);
  const lo=sAllP[Math.floor(sAllP.length*0.1)]||top5[0]?.value||0;
  const hi=sAllP[Math.floor(sAllP.length*0.9)]||top5[0]?.value||0;
  const topVal=top5[0]?.value;
  const tempSignals=Object.entries(details).filter(([,d])=>d.type==="temporal"&&ok(d.pred)).map(([name,d])=>({name,pred:M.mod(Math.round(d.pred)),w:d.w,match:topVal!=null&&M.mod(Math.round(d.pred))===topVal}));
  const tempAgree=tempSignals.filter(s=>s.match).length;
  const tempTotal=tempSignals.length;
  const dateSigList=Object.entries(details).filter(([,d])=>(d.type==="date"||d.type==="patternbank")&&ok(d.pred)).map(([name,d])=>({name,pred:M.mod(Math.round(d.pred)),w:d.w,match:topVal!=null&&M.mod(Math.round(d.pred))===topVal}));
  const dateAgree=dateSigList.filter(s=>s.match).length;
  const dateTotal=dateSigList.length;
  perf.totalMs=PERF_NOW()-tStart;
  return{top5,details,consensus,algoCount:ac,conf,confClr,variance:allP.length>1?+M.std(allP).toFixed(1):0,regime,bandLo:lo,bandHi:hi,tempSignals,tempAgree,tempTotal,dateSigList,dateAgree,dateTotal,familyAgreement:top1FamSupport,perf,algoBudget,lightweight};
}

// ── NEURAL RUNNING SCORE (per-algo accuracy tracker) ──
// Tiered rewards + streak multiplier: consecutive hits compound.
function updateNeuralScores(pred,actual,prevScores,regime){
  const next={...prevScores};
  if(!pred)return next;
  // Adaptive learning rate by regime
  const alpha=regime==="volatile"?0.35:regime==="flat"?0.12:0.20;
  // Streak counters stored with _ prefix
  Object.entries(pred.details).forEach(([name,info])=>{
    if(!ok(info.pred))return;
    const p=M.mod(Math.round(info.pred));
    const ex=p===actual;
    const nr=!ex&&M.nearR(p,actual,regime);
    const close=!ex&&!nr&&M.cd(p,actual)<=5;
    // Tiered reward: exact=+3, near=+1, close=+0.2, miss=-0.6
    const baseReward=ex?3:nr?1:close?0.2:-0.6;
    // Streak tracking per algo
    const streakKey="_s_"+name;
    const curStreak=next[streakKey]||0;
    const newStreak=ex?(curStreak>=0?curStreak+1:1):nr?Math.max(0,curStreak):(curStreak<=0?curStreak-1:-1);
    next[streakKey]=Math.max(-5,Math.min(8,newStreak));
    // Streak multiplier: up to 1.6x boost for 4+ consecutive hits
    const streakMult=newStreak>=4?1.6:newStreak>=2?1.3:newStreak<=-3?0.7:1.0;
    const reward=baseReward*streakMult;
    const prev=next[name]!=null?next[name]:0;
    next[name]=+((1-alpha)*prev+alpha*reward).toFixed(3);
  });
  return next;
}
function updateAlgoLeaderboard(lb,col,pred){
  const next={...lb,[col]:{...(lb[col]||{})}};
  if(!pred||!pred.details)return next;
  const winners=Object.entries(pred.details).sort((a,b)=>(b[1]?.w||0)-(a[1]?.w||0)).slice(0,8);
  const decay=0.995;
  const minKeep=0.0005;
  Object.keys(next[col]).forEach(name=>{
    next[col][name]=+(next[col][name]*decay).toFixed(4);
    if(next[col][name]<minKeep)delete next[col][name];
  });
  winners.forEach(([name,info],i)=>{
    const gain=(info?.w||0)/(i+1);
    next[col][name]=+(((next[col][name]||0)*0.9)+gain*0.1).toFixed(4);
  });
  return next;
}

// ── CALIBRATION TRACKER ─────────────────────────
function updateCalibration(conf,wasExact,prevCal){
  const cal={...prevCal};
  if(!cal[conf])cal[conf]={right:0,total:0};
  cal[conf].total++;
  if(wasExact)cal[conf].right++;
  return cal;
}
function getCalibrationLabel(conf,cal){
  if(!cal||!cal[conf]||cal[conf].total<3)return conf;
  const rate=Math.round(cal[conf].right/cal[conf].total*100);
  return conf+"("+rate+"%)";
}

// Fix 4+9: updateW now takes regime+calibration, maintains perRegime weights
function updateW(pred,actual,W,predRow,regime,calibration){
  const gw={...W.global||{}};
  const rw={...W.perRow||{}};
  const rnw={...W.perRange||{}};
  const rgw={...(W.perRegime||{})};
  if(!rgw[regime])rgw[regime]={};
  if(!pred)return{global:gw,perRow:rw,perRange:rnw,perRegime:rgw};
  const cr=Math.floor(actual/25);
  if(!rw[predRow])rw[predRow]={};
  if(!rnw[cr])rnw[cr]={};
  const calMult=calibration?getCalibMult(pred.conf,calibration):1.0;
  Object.entries(pred.details).forEach(([name,info])=>{
    if(!ok(info.pred))return;
    const p=M.mod(Math.round(info.pred));
    const ex=p===actual,nr=!ex&&M.near(p,actual,2);
    const mult=ex?1.4:nr?1.1:0.80;
    const mom=gw["_m_"+name]!=null?gw["_m_"+name]:1.0;
    const newMom=0.75*mom+0.25*mult;
    gw["_m_"+name]=Math.min(2.0,Math.max(0.3,newMom));
    const prev=gw[name]!=null?gw[name]:1;
    const decay=prev>1?0.96:0.98;
    // Cold-streak dampening: if momentum < 0.85 for 2+ misses, increase decay speed
    const coldDamp=newMom<0.82?0.92:decay;
    gw[name]=Math.min(6,Math.max(0.04,prev*newMom))*coldDamp+(1-coldDamp);
    const rp=rw[predRow][name]!=null?rw[predRow][name]:1;
    rw[predRow][name]=Math.min(5,Math.max(0.05,rp*(ex?1.5:nr?1.15:0.75)));
    const rn=rnw[cr][name]!=null?rnw[cr][name]:1;
    rnw[cr][name]=Math.min(5,Math.max(0.05,rn*mult));
    // Fix 4: per-regime weight track
    const rv=rgw[regime][name]!=null?rgw[regime][name]:1;
    rgw[regime][name]=Math.min(5,Math.max(0.05,rv*(ex?1.45*calMult:nr?1.1:0.82)));
  });
  // Fix 8: per-DOW and per-lunar weight update (passed via extra param from checkAndLearn)
  return{global:gw,perRow:rw,perRange:rnw,perRegime:rgw,
    neuralScores:updateNeuralScores(pred,actual,W.neuralScores||{},regime)};
}

// Fix 8: update date-conditioned weights after learning
function updateDateWeights(pred,actual,W,dateCtx){
  if(!pred||!dateCtx)return W;
  const next={...W};
  const {dow,lunar}=dateCtx;
  const perDOW={...(W.perDOW||{})};
  const perLunar={...(W.perLunar||{})};
  if(!perDOW[dow])perDOW[dow]={};
  if(!perLunar[lunar])perLunar[lunar]={};
  Object.entries(pred.details).forEach(([name,info])=>{
    if(!ok(info.pred))return;
    const p=M.mod(Math.round(info.pred));
    const ex=p===actual,nr=!ex&&M.near(p,actual,2);
    const mult=ex?1.3:nr?1.1:0.85;
    const d=perDOW[dow][name]!=null?perDOW[dow][name]:1;
    perDOW[dow][name]=Math.min(4,Math.max(0.1,d*mult));
    const l=perLunar[lunar][name]!=null?perLunar[lunar][name]:1;
    perLunar[lunar][name]=Math.min(4,Math.max(0.1,l*mult));
  });
  next.perDOW=perDOW;
  next.perLunar=perLunar;
  return next;
}

// ── ADAPTIVE ALGO GENERATOR ────────────────────
function generateAlgos(data,existing){
  const out=[];
  COLS.forEach(col=>{
    const s=getSeries(col,data),n=s.length;
    if(n<6)return;

    // ── 1. best linear: a*s[i-1]+b mod 100 ──
    let bl={sc:-1,a:1,b:0};
    for(let a=1;a<10;a++)for(let b=0;b<100;b+=3){
      let sc=0;
      for(let i=1;i<n;i++){if(M.mod(a*s[i-1]+b)===s[i])sc++;}
      if(sc>bl.sc)bl={sc,a,b};
    }
    if(bl.sc>1){const nm="Lin_"+col+"_"+bl.a+"x"+bl.b;if(!existing.has(nm))out.push({name:nm,code:"(s,M)=>[M.mod("+bl.a+"*s[s.length-1]+"+bl.b+")]",desc:"Auto linear col "+col,enabled:true,generated:true,createdAt:Date.now()});}

    // ── 2. best cyclic: exact period detection with recency ──
    let bc={sc:-1,p:2};
    for(let p=2;p<=10;p++){
      let sc=0;
      for(let i=p;i<n;i++){
        const rw=i>=n-4?2.0:1.0;
        if(M.cd(s[i],s[i-p])<=2)sc+=rw;
      }
      if(sc/(n-p)>bc.sc)bc={sc:sc/(n-p),p};
    }
    if(bc.sc>0.35){const nm="Cyc_"+col+"_p"+bc.p;if(!existing.has(nm))out.push({name:nm,code:"(s,M)=>{const n=s.length,p="+bc.p+",b=n%p||p;return[M.mod(s[n-b]||s[n-1])];}",desc:"Auto cyclic p"+bc.p+" col "+col,enabled:true,generated:true,createdAt:Date.now()});}

    // ── 3. avg gap (recent-biased) ──
    const gaps=[];for(let i=1;i<n;i++){let g=s[i]-s[i-1];if(g>50)g-=100;if(g<-50)g+=100;gaps.push(g);}
    const avgGap=Math.round(M.mean(gaps.slice(-6)));
    if(avgGap!==0){const nm="Gap_"+col+"_"+(avgGap>0?"+":"")+avgGap;if(!existing.has(nm))out.push({name:nm,code:"(s,M)=>[M.mod(s[s.length-1]+("+avgGap+"))]",desc:"Auto gap "+(avgGap>0?"+":"")+avgGap+" col "+col,enabled:true,generated:true,createdAt:Date.now()});}

    // ── 4. crossover: blend top-2 built-in predictions DYNAMICALLY ──
    const scoredBI=Object.entries(A).map(([nm,fn])=>({nm,sc:btScore(fn,s)})).sort((a,b)=>b.sc-a.sc);
    if(scoredBI.length>=2&&scoredBI[0].sc>0.15){
      try{
        // Store names not fixed offset — code is re-evaluated each prediction
        const n1=scoredBI[0].nm,n2=scoredBI[1].nm;
        const nm="Cross_"+col+"_"+n1.slice(0,4)+"_"+n2.slice(0,4);
        if(!existing.has(nm))out.push({name:nm,
          code:"(s,M)=>{try{const A="+JSON.stringify({[n1]:true,[n2]:true})+";const p1=("+A[n1].toString()+")(s)[0],p2=("+A[n2].toString()+")(s)[0];return[M.mod(Math.round((p1+p2)/2))];}catch(e){return[s[s.length-1]];}}",
          desc:"Auto crossover "+n1+"+"+n2+" col "+col,enabled:true,generated:true,createdAt:Date.now()});
      }catch(e){}
    }

    // ── 5. digit-sum chain ──
    if(n>=5){
      let dsb={sc:-1,k:1,c:0};
      for(let k=1;k<=11;k++)for(let c=0;c<100;c+=5){
        let sc=0;for(let i=1;i<n;i++)if(M.mod(M.ds(s[i-1])*k+c)===s[i])sc++;
        if(sc>dsb.sc)dsb={sc,k,c};
      }
      if(dsb.sc>=2){const nm="DsChain_"+col+"_k"+dsb.k+"c"+dsb.c;
        if(!existing.has(nm))out.push({name:nm,code:"(s,M)=>[M.mod(M.ds(s[s.length-1])*"+dsb.k+"+"+dsb.c+")]",desc:"Auto digit-sum chain col "+col,enabled:true,generated:true,createdAt:Date.now()});}
    }

    // ── 6. two-step recurrence ──
    if(n>=6){
      let r2b={sc:-1,a:1,b:0,c:0};
      for(const a of[1,2,-1,3])for(const b of[0,1,-1,2])for(const c of[0,3,7,11,-3,-7]){
        let sc=0;for(let i=2;i<n;i++)if(M.mod(a*s[i-1]+b*s[i-2]+c)===s[i])sc++;
        if(sc>r2b.sc)r2b={sc,a,b,c};
      }
      if(r2b.sc>=3){const nm="Rec2_"+col+"_a"+r2b.a+"b"+r2b.b;
        if(!existing.has(nm))out.push({name:nm,code:"(s,M)=>{const n=s.length;return[M.mod("+(r2b.a)+"*s[n-1]+"+(r2b.b)+"*s[n-2]+"+(r2b.c)+")];}",desc:"Auto 2-step recurrence col "+col,enabled:true,generated:true,createdAt:Date.now()});}
    }

    // ── 7. modular step ──
    if(n>=5){
      let msb={sc:-1,a:1,c:1,m:97};
      for(const a of[2,3,5,7,11])for(const c of[1,3,7,13,17])for(const m of[97,89,83,79]){
        let sc=0;for(let i=1;i<n;i++)if(M.mod((a*s[i-1]+c)%m)===s[i])sc++;
        if(sc>msb.sc)msb={sc,a,c,m};
      }
      if(msb.sc>=2){const nm="ModStep_"+col+"_a"+msb.a+"m"+msb.m;
        if(!existing.has(nm))out.push({name:nm,code:"(s,M)=>[M.mod(("+msb.a+"*s[s.length-1]+"+msb.c+")%"+msb.m+")]",desc:"Auto modular step col "+col,enabled:true,generated:true,createdAt:Date.now()});}
    }

    // ── 8. NEW: XOR-lag detector — s[i] = s[i-j] XOR s[i-k] ──
    if(n>=6){
      let xb={sc:-1,j:2,k:1};
      for(let j=2;j<=Math.min(6,n-1);j++)for(let k=1;k<j;k++){
        let sc=0;for(let i=j;i<n;i++)if((s[i-j]^s[i-k])===s[i]||(M.mod(s[i-j]^s[i-k])===s[i]))sc++;
        if(sc>xb.sc)xb={sc,j,k};
      }
      if(xb.sc>=3){const nm="XorLag_"+col+"_j"+xb.j+"k"+xb.k;
        if(!existing.has(nm))out.push({name:nm,code:"(s,M)=>{const n=s.length;return[M.mod(s[n-"+xb.j+"]^s[n-"+xb.k+"])];}",desc:"Auto XOR-lag col "+col,enabled:true,generated:true,createdAt:Date.now()});}
    }

    // ── 9. NEW: mod-period shift — s[i] = (s[i-p] + k) mod 100 ──
    if(n>=6){
      let mpb={sc:-1,p:2,k:0};
      for(let p=2;p<=Math.min(8,n-1);p++)for(let k=0;k<100;k+=2){
        let sc=0;for(let i=p;i<n;i++)if(M.mod(s[i-p]+k)===s[i])sc++;
        if(sc>mpb.sc)mpb={sc,p,k};
      }
      if(mpb.sc>=3){const nm="MPShift_"+col+"_p"+mpb.p+"k"+mpb.k;
        if(!existing.has(nm))out.push({name:nm,code:"(s,M)=>[M.mod(s[s.length-"+mpb.p+"]+"+(mpb.k)+")]",desc:"Auto mod-period shift col "+col,enabled:true,generated:true,createdAt:Date.now()});}
    }

    // ── 10. NEW: complement-chain — s[i] = (100 - s[i-1]*a + c) mod 100 ──
    if(n>=5){
      let ccb={sc:-1,a:1,c:0};
      for(const a of[1,2,3])for(let c=0;c<100;c+=5){
        let sc=0;for(let i=1;i<n;i++)if(M.mod(100-a*s[i-1]+c)===s[i])sc++;
        if(sc>ccb.sc)ccb={sc,a,c};
      }
      if(ccb.sc>=3){const nm="CompChain_"+col+"_a"+ccb.a+"c"+ccb.c;
        if(!existing.has(nm))out.push({name:nm,code:"(s,M)=>[M.mod(100-"+ccb.a+"*s[s.length-1]+"+ccb.c+")]",desc:"Auto complement-chain col "+col,enabled:true,generated:true,createdAt:Date.now()});}
    }
  });
  return out.slice(0,32); // slightly larger batch per run
}
// Bug 11 fix: hard cap enforced at generate time + prune time
const MAX_GENERATED_ALGOS=60;

// ── TOURNAMENT (Fix 5: real mutation) ─────────────
function mutateCode(code){
  // Replace numeric constants with slightly perturbed versions
  let mutated=code;
  const numRe=/(\d+\.?\d*)/g;
  const nums=[];
  let m;
  while((m=numRe.exec(code))!==null){
    const v=parseFloat(m[1]);
    // Only mutate small constants (coefficients), not indices/lengths
    if(v>0&&v<200&&v!==0&&String(v).length<=5)nums.push({idx:m.index,len:m[0].length,v});
  }
  if(!nums.length)return code;
  // Pick 1-3 random constants to mutate
  const toMutate=nums.sort(()=>Math.random()-0.5).slice(0,Math.min(3,nums.length));
  // Apply from end to start to preserve indices
  toMutate.sort((a,b)=>b.idx-a.idx);
  toMutate.forEach(({idx,len,v})=>{
    // Perturb by ±5-25% with some rounding
    const delta=v*(0.05+Math.random()*0.20)*(Math.random()<0.5?1:-1);
    const newV=Math.round((v+delta)*100)/100;
    if(newV>0)mutated=mutated.slice(0,idx)+String(newV)+mutated.slice(idx+len);
  });
  return mutated;
}
function runTournament(customs,data,weights){
  const scored=customs.map(ca=>{
    let tot=0,wt=0;
    COLS.forEach(col=>{
      const s=getSeries(col,data);
      if(s.length<5)return;
      const fn=makeCustomFn(ca.code);
      if(!fn)return;
      const bt=btScore(fn,s);
      // Include walkFwd and neural score in tournament (mirrors actual voting)
      let wfBoost=1.0;
      if(s.length>=8){const wf=walkFwd(fn,s);if(wf&&wf.total>=2){const wr=(wf.exact+wf.near*0.4)/wf.total;wfBoost=0.6+wr*0.8;}}
      const ns=weights&&weights[col]&&weights[col].neuralScores&&weights[col].neuralScores[ca.name]!=null?weights[col].neuralScores[ca.name]:0;
      const nBoost=ns>1?1.4:ns>0?1.1:ns<-0.5?0.7:1.0;
      tot+=(0.2+Math.sqrt(bt)*3.5)*wfBoost*nBoost;
      wt++;
    });
    return{...ca,_sc:wt?tot/wt:0};
  }).sort((a,b)=>b._sc-a._sc);
  if(scored.length<4)return customs;
  const half=Math.ceil(scored.length/2);
  const top=scored.slice(0,half);
  // Fix 5: generate real mutants by perturbing constants in parent code
  const mutants=scored.slice(half).map((_,i)=>{
    const parent=top[i%top.length];
    const mutCode=mutateCode(parent.code);
    // Verify mutant actually parses
    const fn=makeCustomFn(mutCode);
    const validCode=fn?mutCode:parent.code;
    const suffix=Date.now()%10000;
    return{...parent,name:"Mut_"+parent.name.slice(-6)+"_"+suffix,
      code:validCode,desc:"Mutant of "+parent.name,generated:true,createdAt:Date.now(),_sc:undefined};
  });
  return[...top,...mutants];
}

// ── AUTO-GENERATE TRIGGER ──────────────────────
function shouldAutoGenerate(rows,genN,lastAutoGenRows,currentAlgoCount){
  if(rows<4)return false;
  if((currentAlgoCount||0)>=MAX_GENERATED_ALGOS)return false;
  const rowsSinceLast=rows-(lastAutoGenRows||0);
  // Generate more aggressively early on, slow down as algo pool fills
  const interval=currentAlgoCount<20?2:currentAlgoCount<40?3:5;
  return rowsSinceLast>=interval;
}
// ── AUTO-PRUNE TRIGGER ──────────────────────────
// Prune on every row add when pool is large enough, not just checkAndLearn
function shouldAutoPrune(customs,rows){
  const gen=(customs||[]).filter(a=>a.generated);
  return gen.length>=8&&rows>=6;
}

// scoreAlgo: optional btCache avoids redundant btScore recomputation during auto-train prune
function scoreAlgo(ca,weights,rows,btCache){
  if(!ca.generated)return Infinity;
  if(ca.benched)return -0.1;
  const fn=makeCustomFn(ca.code);
  if(!fn)return -999;
  let totalNs=0,totalStreak=0,nsCount=0,totalBt=0,totalWf=0,btCnt=0;
  COLS.forEach(col=>{
    const ns=weights[col]&&weights[col].neuralScores?weights[col].neuralScores:{};
    if(ns[ca.name]!=null){totalNs+=ns[ca.name];nsCount++;}
    const sk=ns["_s_"+ca.name];if(sk!=null)totalStreak+=sk;
    const s=getSeries(col,rows);
    if(s.length>=5){
      const cached=btCache&&btCache[col+"__"+ca.name];
      if(cached){
        totalBt+=cached.bt;
        totalWf+=(cached.wfBoost-0.6)/0.8; // approximate wf rate from boost
      } else {
        totalBt+=btScore(fn,s);
        if(s.length>=8){const wf=walkFwd(fn,s);if(wf&&wf.total>=2)totalWf+=(wf.exact+wf.near*0.4)/wf.total;}
      }
      btCnt++;
    }
  });
  const avgNs=nsCount?totalNs/nsCount:0;
  const avgStreak=nsCount?totalStreak/nsCount:0;
  const avgBt=btCnt?totalBt/btCnt:0.05;
  const avgWf=btCnt?totalWf/btCnt:0.05;
  return avgNs*0.40+avgStreak*0.20+avgBt*2.0+avgWf*1.2;
}

// Fix 10: redundancy detection — find algos that always predict same value
function detectRedundant(customs,rows){
  if(customs.length<4)return new Set();
  const redundant=new Set();
  const predictions={};
  customs.forEach(ca=>{
    if(!ca.generated)return;
    const fn=makeCustomFn(ca.code);
    if(!fn)return;
    const key=COLS.map(col=>{
      const s=getSeries(col,rows);
      if(s.length<3)return"?";
      try{return fn(s).join(",");}catch(e){return"?";}
    }).join("|");
    if(predictions[key]){
      // Duplicate: keep the one with higher score
      redundant.add(ca.name);
    } else {
      predictions[key]=ca.name;
    }
  });
  return redundant;
}

function pruneWeakAlgos(customs,weights,rows,btCache){
  if(!customs||customs.length===0)return{pruned:customs,removed:[]};
  const generated=customs.filter(ca=>ca.generated);
  const userDefined=customs.filter(ca=>!ca.generated);
  if(generated.length<2)return{pruned:customs,removed:[]};
  if(rows.length<4)return{pruned:customs,removed:[]};
  const redundant=detectRedundant(generated,rows);
  const scored=generated.map((ca,idx)=>({ca,score:scoreAlgo(ca,weights,rows,btCache),origIdx:idx}));
  scored.sort((a,b)=>a.score-b.score); // ascending: worst first

  // Pre-build index map — O(1) lookup, fixes O(n²) bug
  const scoreRank=new Map(scored.map((x,i)=>[x.ca.name,i]));

  const overCap=Math.max(0,generated.length-MAX_GENERATED_ALGOS);
  // Dynamic threshold: scales with data — more rows = higher bar
  const threshold=rows.length>=25?0.15:rows.length>=15?0.08:rows.length>=10?0.00:-0.50;
  const benchThreshold=threshold+0.12;

  const removed=[];
  const benched=[];
  const kept=[];

  scored.forEach(({ca,score})=>{
    // Always remove redundant
    if(redundant.has(ca.name)){
      removed.push({name:ca.name,reason:"redundant"});
      return;
    }
    // Remove over-cap (lowest ranked first) — use pre-built rank map
    const rank=scoreRank.get(ca.name)??999;
    if(overCap>0&&rank<overCap&&removed.length<overCap+3){
      removed.push({name:ca.name,reason:"over_cap"});
      return;
    }
    // Remove hard failures (max 4 per cycle to avoid mass deletion)
    if(score<threshold&&removed.length<4){
      removed.push({name:ca.name,reason:"score:"+score.toFixed(2)});
      return;
    }
    // Bench borderline algos
    if(score<benchThreshold&&!ca.benched){
      benched.push({...ca,enabled:false,benched:true,benchedAt:Date.now()});
      return;
    }
    // Re-enable benched algos that recovered
    if(ca.benched&&score>=benchThreshold){
      kept.push({...ca,enabled:true,benched:false});
      return;
    }
    kept.push(ca);
  });

  const pruned=[...userDefined,...kept,...benched];
  return{pruned,removed};
}

// ── EXPORT HELPERS ─────────────────────────────
// Bug 9 fix: CSV now includes Date column so re-import preserves all date metadata
function doExportCSV(data,preds,predRow){
  const rows=data.map(r=>pad2(r.row)+","+(r.A!=null?r.A:"XX")+","+(r.B!=null?r.B:"XX")+","+(r.C!=null?r.C:"XX")+","+(r.D!=null?r.D:"XX")+","+(r.date||"")).join("\n");
  let pLine="";
  if(preds&&predRow){
    const pa=preds.A?preds.A.top5[0]?.value:null,pb=preds.B?preds.B.top5[0]?.value:null;
    const pc=preds.C?preds.C.top5[0]?.value:null,pd=preds.D?preds.D.top5[0]?.value:null;
    pLine="\n"+pad2(predRow)+","+(ok(pa)?pad2(pa):"?")+","+(ok(pb)?pad2(pb):"?")+","+(ok(pc)?pad2(pc):"?")+","+(ok(pd)?pad2(pd):"?")+",(PRED)";
  }
  const blob=new Blob(["Row,A,B,C,D,Date\n"+rows+pLine],{type:"text/csv"});
  const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="ape-v13-"+Date.now()+".csv";document.body.appendChild(a);a.click();document.body.removeChild(a);URL.revokeObjectURL(url);
}
function doExportJSON(state){
  const blob=new Blob([JSON.stringify(state,null,2)],{type:"application/json"});
  const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="ape-v13-backup-"+Date.now()+".json";document.body.appendChild(a);a.click();document.body.removeChild(a);URL.revokeObjectURL(url);
}

// ── WEIGHTS IMPORT/EXPORT ─────────────────────
function doExportWeights(state){
  const payload={
    version:"ape-v13-weights",
    exportedAt:new Date().toISOString(),
    weights:state.weights,
    customs:state.customs,
    accLog:state.accLog,
  };
  const blob=new Blob([JSON.stringify(payload,null,2)],{type:"application/json"});
  const url=URL.createObjectURL(blob),a=document.createElement("a");
  a.href=url;a.download="ape-v13-weights-"+Date.now()+".json";
  document.body.appendChild(a);a.click();document.body.removeChild(a);URL.revokeObjectURL(url);
}
function migrateWeights(weights){
  // Bug 8 fix: ensure all new weight table fields exist on import
  const migrated={};
  COLS.forEach(col=>{
    const w=weights&&weights[col]?{...weights[col]}:{};
    if(!w.global)w.global={};
    if(!w.perRow)w.perRow={};
    if(!w.perRange)w.perRange={};
    if(!w.perRegime)w.perRegime={};
    if(!w.perDOW)w.perDOW={};
    if(!w.perLunar)w.perLunar={};
    if(!w.neuralScores)w.neuralScores={};
    migrated[col]=w;
  });
  return migrated;
}
function parseImportedWeights(parsed){
  if(parsed.version==="ape-v13-weights"||parsed.version==="ape-v7-weights"){
    return{weights:migrateWeights(parsed.weights),customs:parsed.customs||[],accLog:parsed.accLog||[]};
  }
  if(parsed.weights&&parsed.customs!==undefined){
    return{weights:migrateWeights(parsed.weights),customs:parsed.customs,accLog:parsed.accLog||[]};
  }
  return null;
}

// ══════════════════════════════════════════════════════════════════════
// ── PATTERN BANK ─ Long-term cross-period memory ──────────────────────
// Distills proven patterns from ALL historical data into a compact structure
// that survives dataset switches, period gaps (2025→2026), and weight resets.
// Updated after every checkAndLearn. Used inside predictCol as a high-weight
// signal layer alongside date signals.
// ══════════════════════════════════════════════════════════════════════

function updatePatternBank(prevBank,col,allDatasets,accLog){
  const bank={...((prevBank||{})[col]||{})};

  // ── 1. Monthly profiles — per-month mean/std across ALL years ──────
  const monthly={};
  Object.values(allDatasets||{}).forEach(ds=>{
    (ds.rows||[]).forEach(r=>{
      if(!ok(r[col])||!r.date)return;
      const pd=parseDate(r.date);if(!pd)return;
      (monthly[pd.m]=monthly[pd.m]||[]).push(r[col]);
    });
  });
  const monthlyProfiles={};
  Object.entries(monthly).forEach(([m,vals])=>{
    if(vals.length>=2)monthlyProfiles[m]={mean:Math.round(M.mean(vals)),std:+M.std(vals).toFixed(1),n:vals.length,
      // Tight = std<6 → near-deterministic for this month
      tight:M.std(vals)<6&&vals.length>=3};
  });
  bank.monthlyProfiles=monthlyProfiles;

  // ── 2. Day-of-month profiles — same calendar day across ALL months/years ──
  const dayMap2={};
  Object.values(allDatasets||{}).forEach(ds=>{
    (ds.rows||[]).forEach(r=>{
      if(!ok(r[col])||!r.date)return;
      const pd=parseDate(r.date);if(!pd)return;
      (dayMap2[pd.d]=dayMap2[pd.d]||[]).push(r[col]);
    });
  });
  const dayProfiles={};
  Object.entries(dayMap2).forEach(([d,vals])=>{
    if(vals.length>=2)dayProfiles[d]={mean:Math.round(M.mean(vals)),std:+M.std(vals).toFixed(1),n:vals.length,
      tight:M.std(vals)<5&&vals.length>=3,
      snug:M.std(vals)<10&&vals.length>=3};
  });
  bank.dayProfiles=dayProfiles;

  // ── 3. Month×DayOfMonth joint profiles (finest granularity) ─────────
  const mdMap={};
  Object.values(allDatasets||{}).forEach(ds=>{
    (ds.rows||[]).forEach(r=>{
      if(!ok(r[col])||!r.date)return;
      const pd=parseDate(r.date);if(!pd)return;
      const k=pd.m+"_"+pd.d;
      (mdMap[k]=mdMap[k]||[]).push(r[col]);
    });
  });
  const mdProfiles={};
  Object.entries(mdMap).forEach(([k,vals])=>{
    if(vals.length>=2)mdProfiles[k]={mean:Math.round(M.mean(vals)),std:+M.std(vals).toFixed(1),n:vals.length,
      tight:M.std(vals)<5};
  });
  bank.mdProfiles=mdProfiles;

  // ── 4. Year-over-Year deltas per month ───────────────────────────────
  // Used to project 2026 from 2025: 2026_mean ≈ 2025_mean + avg_annual_delta
  const yoyByMonthYear={};
  Object.values(allDatasets||{}).forEach(ds=>{
    (ds.rows||[]).forEach(r=>{
      if(!ok(r[col])||!r.date)return;
      const pd=parseDate(r.date);if(!pd)return;
      const k=pd.m+"_"+pd.y;
      (yoyByMonthYear[k]=yoyByMonthYear[k]||{m:pd.m,y:pd.y,vals:[]}).vals.push(r[col]);
    });
  });
  const yoyDeltas={};
  const monthYearMeans={};
  Object.values(yoyByMonthYear).forEach(({m,y,vals})=>{
    if(!monthYearMeans[m])monthYearMeans[m]={};
    monthYearMeans[m][y]=Math.round(M.mean(vals));
  });
  Object.entries(monthYearMeans).forEach(([m,yearMap])=>{
    const years=Object.keys(yearMap).map(Number).sort();
    if(years.length>=2){
      const deltas=[];
      for(let i=1;i<years.length;i++)deltas.push(yearMap[years[i]]-yearMap[years[i-1]]);
      yoyDeltas[m]={yearMap,avgDelta:Math.round(M.mean(deltas)),lastYear:years[years.length-1],lastMean:yearMap[years[years.length-1]]};
    }
  });
  bank.yoyDeltas=yoyDeltas;

  // ── 5. Top algo hit-rates from accLog (cross-session truth) ──────────
  const algoHits={},algoCounts={};
  (accLog||[]).forEach(entry=>{
    const details=entry.algoDetails?.[col];
    const actual=entry.actuals?.[col];
    if(!ok(actual)||!details)return;
    Object.entries(details).forEach(([name,pred])=>{
      if(!ok(pred))return;
      if(!algoCounts[name]){algoCounts[name]=0;algoHits[name]=0;}
      algoCounts[name]++;
      if(M.mod(Math.round(pred))===actual)algoHits[name]++;
    });
  });
  bank.topAlgos=Object.entries(algoCounts)
    .filter(([,n])=>n>=3)
    .map(([name,n])=>({name,hitRate:+(algoHits[name]/n).toFixed(3),n}))
    .sort((a,b)=>b.hitRate-a.hitRate).slice(0,20);

  // ── 6. Periodic pattern strength ─────────────────────────────────────
  const globalSer=getGlobalSeries(col,allDatasets);
  if(globalSer.length>=12){
    let bestPer={p:7,strength:0};
    for(let p=2;p<=14;p++){
      let sc=0;
      for(let i=p;i<globalSer.length;i++)sc+=Math.max(0,1-M.cd(globalSer[i],globalSer[i-p])/15);
      const norm=sc/(globalSer.length-p);
      if(norm>bestPer.strength)bestPer={p,strength:+norm.toFixed(3)};
    }
    bank.bestPeriod=bestPer;
  }

  bank.updatedAt=Date.now();
  return bank;
}

// ── Signal generator from Pattern Bank ──────────────────────────────────
// Returns prediction signals based on stored patterns (month/day/yoy profiles)
function getPatternBankSignals(col,patternBank,targetDate){
  const res={};
  const pb=patternBank?.[col];
  if(!pb)return res;
  const td=targetDate?parseDate(targetDate):null;

  if(td){
    // Month×Day exact profile — finest grain, highest confidence
    const mdKey=td.m+"_"+td.d;
    const mdP=pb.mdProfiles?.[mdKey];
    if(mdP&&mdP.n>=2){
      res["PB_MDExact"]=[M.mod(mdP.mean)];
      if(mdP.tight)res["PB_MDTight"]=[M.mod(mdP.mean)]; // extra high-weight duplicate
    }

    // Day-of-month profile
    const dayP=pb.dayProfiles?.[td.d];
    if(dayP&&dayP.n>=2){
      res["PB_DayProfile"]=[M.mod(dayP.mean)];
      if(dayP.tight)res["PB_DayTight"]=[M.mod(dayP.mean)];
      if(dayP.snug&&!dayP.tight)res["PB_DaySnug"]=[M.mod(dayP.mean)];
    }

    // Monthly profile
    const moP=pb.monthlyProfiles?.[td.m];
    if(moP&&moP.n>=2){
      res["PB_MonthProfile"]=[M.mod(moP.mean)];
      if(moP.tight)res["PB_MonthTight"]=[M.mod(moP.mean)];
    }

    // Year-over-Year projection — predict 2026 from 2025 trend
    const yoy=pb.yoyDeltas?.[td.m];
    if(yoy&&td.y>yoy.lastYear){
      const yearGap=td.y-yoy.lastYear;
      const yoyPred=M.mod(Math.round(yoy.lastMean+yoy.avgDelta*yearGap));
      res["PB_YoY"]=[yoyPred];
      // Also YoY + day correction (if day profile exists)
      if(dayP&&dayP.n>=2){
        const dayOffset=dayP.mean-moP?.mean||0;
        res["PB_YoYDay"]=[M.mod(Math.round(yoyPred+dayOffset*0.5))];
      }
    }
  }

  // Best periodic pattern from history
  if(pb.bestPeriod&&pb.bestPeriod.strength>0.45){
    // This doesn't use date but uses the period structure
    res["PB_BestPeriod"]=[null]; // populated by caller with series context
  }

  return res;
}
const SK="ape-v13";
// Debounced save: batches rapid state changes (e.g. during predict) into one write
let _saveTimer=null;
let _saveWarned=false;
function saveS(s){
  clearTimeout(_saveTimer);
  _saveTimer=setTimeout(()=>{
    try{
      const str=JSON.stringify(s);
      // Warn user if approaching storage limit (>3MB)
      if(str.length>3*1024*1024&&!_saveWarned){
        _saveWarned=true;
        console.warn("APE: state size "+Math.round(str.length/1024)+"KB — consider exporting weights to free space");
      }
      localStorage.setItem(SK,str);
    }catch(e){
      // Storage full — notify via console (don't crash)
      console.error("APE: localStorage save failed (storage full?)",e.message);
    }
  },300);
}
async function loadS(){try{const r=localStorage.getItem(SK);return r?JSON.parse(r):null;}catch(e){return null;}}
function fresh(){
  return{
    datasets:{def:{name:"Dataset 1",rows:[]}},
    active:"def",
    weights:{A:{global:{},perRow:{},perRange:{},perRegime:{},perDOW:{},perLunar:{},neuralScores:{}},B:{global:{},perRow:{},perRange:{},perRegime:{},perDOW:{},perLunar:{},neuralScores:{}},C:{global:{},perRow:{},perRange:{},perRegime:{},perDOW:{},perLunar:{},neuralScores:{}},D:{global:{},perRow:{},perRange:{},perRegime:{},perDOW:{},perLunar:{},neuralScores:{}}},
    algoLeaderboard:{A:{},B:{},C:{},D:{}},
    calibration:{A:{},B:{},C:{},D:{}},
    patternBank:{A:{},B:{},C:{},D:{}},
    customs:[],accLog:[],preds:null,predRow:null,predDate:null,genN:0,tourN:0,lastAutoGenRows:0,pruneLog:[]
  };
}

// ── APP ────────────────────────────────────────
// ── ERROR BOUNDARY ────────────────────────────
class ErrorBoundary extends React.Component{
  constructor(p){super(p);this.state={err:null};}
  static getDerivedStateFromError(e){return{err:e};}
  componentDidCatch(e,info){console.error("APE Error:",e,info);}
  render(){
    if(this.state.err){
      return React.createElement("div",{style:{background:"#060709",minHeight:"100vh",display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",color:"#f87171",fontFamily:"monospace",gap:16,padding:20}},
        React.createElement("div",{style:{fontSize:32}},"⚠"),
        React.createElement("div",{style:{fontSize:14,fontWeight:700}},"APE encountered an error"),
        React.createElement("div",{style:{fontSize:11,color:"#4a4e6a",maxWidth:400,textAlign:"center"}},(this.state.err.message||"Unknown error")),
        React.createElement("button",{onClick:()=>{localStorage.removeItem("ape-v13");window.location.reload();},style:{background:"#f87171",border:"none",color:"#fff",padding:"8px 20px",borderRadius:6,cursor:"pointer",fontFamily:"monospace",fontSize:12,marginTop:8}},"🔄 Reset & Reload"),
        React.createElement("div",{style:{fontSize:9,color:"#252840"}},"Your data is saved. Only the current view crashed.")
      );
    }
    return this.props.children;
  }
}

function todayStr(){const d=new Date();return d.getFullYear()+"-"+String(d.getMonth()+1).padStart(2,"0")+"-"+String(d.getDate()).padStart(2,"0");}
export default function App(){return React.createElement(ErrorBoundary,null,React.createElement(AppInner,null));}
function AppInner(){
  const[S,setS]=useState(fresh());
  const[loaded,setLoaded]=useState(false);
  const[tab,setTab]=useState("data");
  const[rowIn,setRowIn]=useState("");
  const[vals,setVals]=useState({A:"",B:"",C:"",D:""});
  const[acts,setActs]=useState({A:"",B:"",C:"",D:""});
  const[bulk,setBulk]=useState("");
  const[showBulk,setShowBulk]=useState(false);
  const[expCol,setExpCol]=useState(null);
  const[checkRes,setCheckRes]=useState(null);
  const[msg,setMsg]=useState({t:"Loading…",c:"idle"});
  const[cCode,setCCode]=useState("");
  const[cName,setCName]=useState("");
  const[cErr,setCErr]=useState("");
  const[genMsg,setGenMsg]=useState("");
  const[wfRes,setWfRes]=useState(null);
  const[corrM,setCorrM]=useState(null);
  const[dsName,setDsName]=useState("");
  const[copyMsg,setCopyMsg]=useState("");
  const[autoTrainStatus,setAutoTrainStatus]=useState(null); // {running,progress,total,done,log,result}
  const autoTrainRef=React.useRef(false); // cancellation flag
  const[weightsMsg,setWeightsMsg]=useState("");
  const[showCalib,setShowCalib]=useState(false);
  const[autoGenLog,setAutoGenLog]=useState([]);
  const[sysLog,setSysLog]=useState([{t:Date.now(),lvl:"info",msg:"APE ready — enter day 1 to begin training"}]);
  const[showTerminal,setShowTerminal]=useState(true);
  const sysRef=React.useRef(null);

  // Push to terminal log — capped at 200 entries, auto-scrolls
  const syslog=React.useCallback((msg,lvl="info")=>{
    setSysLog(prev=>{
      const entry={t:Date.now(),lvl,msg};
      const next=[...prev.slice(-199),entry];
      return next;
    });
    setTimeout(()=>{if(sysRef.current)sysRef.current.scrollTop=sysRef.current.scrollHeight;},30);
  },[]);
  const[missingPreds,setMissingPreds]=useState({});
  const[dateIn,setDateIn]=useState(todayStr());
  const[predDate,setPredDate]=useState("");

  const st=(t,c)=>setMsg({t,c:c||"ok"});

  function upd(fn){
    setS(prev=>{
      const next=typeof fn==="function"?fn(prev):{...prev,...fn};
      _TC.bumpVer();
      saveS(next);
      return next;
    });
  }

  const rows=S.datasets&&S.datasets[S.active]?S.datasets[S.active].rows:[];
  function setRows(fn){
    upd(prev=>{
      const r=typeof fn==="function"?fn(prev.datasets[prev.active]?prev.datasets[prev.active].rows:[]):fn;
      const ds={...prev.datasets};
      ds[prev.active]={...ds[prev.active],rows:r};
      return{...prev,datasets:ds};
    });
  }
  useEffect(()=>{
    _TC.bumpVer();
    _TC.clear();
  },[S.datasets,S.active]);

  useEffect(()=>{
    loadS().then(saved=>{
      if(saved){
        if(saved.dataset&&!saved.datasets){saved.datasets={def:{name:"Dataset 1",rows:saved.dataset}};saved.active="def";delete saved.dataset;}
        if(!saved.active)saved.active="def";
        if(!saved.weights||!saved.weights.A||!saved.weights.A.global)saved.weights={A:{global:{},perRow:{},perRange:{},neuralScores:{}},B:{global:{},perRow:{},perRange:{},neuralScores:{}},C:{global:{},perRow:{},perRange:{},neuralScores:{}},D:{global:{},perRow:{},perRange:{},neuralScores:{}}};
        // Migrate all weight tables (uses same logic as import)
        saved.weights=migrateWeights(saved.weights);
        if(!saved.calibration)saved.calibration={A:{},B:{},C:{},D:{}};
        if(!saved.patternBank)saved.patternBank={A:{},B:{},C:{},D:{}};
        if(!saved.algoLeaderboard)saved.algoLeaderboard={A:{},B:{},C:{},D:{}};
        if(!saved.customs)saved.customs=[];
        if(!saved.genN)saved.genN=0;
        if(!saved.tourN)saved.tourN=0;
        setS(saved);
        const n=saved.datasets&&saved.datasets[saved.active]?saved.datasets[saved.active].rows.length:0;
        st("Restored — "+n+" rows · "+(saved.customs?saved.customs.length:0)+" algos · "+(saved.accLog?saved.accLog.length:0)+" sessions");
        setTimeout(()=>syslog("💾 Restored "+n+" rows, "+(saved.customs?saved.customs.length:0)+" algos, "+(saved.accLog?saved.accLog.length:0)+" sessions","info"),100);
      }else{
        st("Welcome to APE — enter Day 1 to start training");
        setTimeout(()=>syslog("🚀 Fresh start — enter day numbers sequentially (1, 2, 3...) with a date for each row","info"),100);
      }
      setLoaded(true);
    });
  },[]);

  function addRow(){
    const r=parseInt(rowIn);
    if(isNaN(r)||r<1||r>9999){st("Day # must be 1–9999","err");return;}
    const entry={row:r};
    if(dateIn&&parseDate(dateIn))entry.date=dateIn;
    for(let i=0;i<COLS.length;i++){
      const col=COLS[i],raw=vals[col].trim();
      if(!raw||raw.toUpperCase()==="XX"){entry[col]=null;continue;}
      const n=parseInt(raw);if(isNaN(n)||n<0||n>99){st(col+" must be 00–99 or XX","err");return;}
      entry[col]=n;
    }
    if(COLS.every(c=>entry[c]===null)){st("Enter at least one value","err");return;}
    setRows(prev=>{
      const updated=[...prev.filter(x=>x.row!==r),entry].sort((a,b)=>a.row-b.row);
      setTimeout(()=>{
        setS(cur=>{
          const curRows=cur.datasets&&cur.datasets[cur.active]?cur.datasets[cur.active].rows:[];
          const _genCount=(cur.customs||[]).filter(a=>a.generated&&!a.benched).length;
          let next=cur;
          // ── Auto-generate new algos ──
          if(shouldAutoGenerate(curRows.length,cur.genN,cur.lastAutoGenRows,_genCount)){
            const existing=new Set([...Object.keys(A),...(cur.customs||[]).map(a=>a.name)]);
            const newAlgos=generateAlgos(curRows,existing);
            if(newAlgos.length>0){
              const log=cur.autoGenLog||[];
              const msg="+"+newAlgos.length+" algos generated ("+curRows.length+" rows)";
              next={...next,customs:[...(next.customs||[]),...newAlgos],genN:(next.genN||0)+1,lastAutoGenRows:curRows.length,autoGenLog:[...log.slice(-9),{at:new Date().toISOString(),msg}]};
              setAutoGenLog(p=>[...p.slice(-4),msg]);
              syslog("🔧 "+msg,"gen");
              // Notify if strong pattern detected
              const strongAlgos=newAlgos.filter(a=>a.name.startsWith("Cyc_")||a.name.startsWith("Rec2_")||a.name.startsWith("XorLag_"));
              if(strongAlgos.length>0)syslog("⚡ Strong pattern detected: "+strongAlgos.map(a=>a.name).join(", ")+" — consider exporting weights!","alert");
            }
          }
          // ── Auto-prune on every row add ──
          if(shouldAutoPrune(next.customs,curRows.length)){
            const {pruned,removed}=pruneWeakAlgos(next.customs||[],next.weights||cur.weights,curRows);
            if(removed.length>0){
              setAutoGenLog(p=>[...p.slice(-(5-Math.min(removed.length,3))),...removed.slice(0,3).map(r=>"Pruned: "+r.name)]);
              removed.forEach(rm=>syslog("🗑 Pruned weak algo: "+rm.name+" ("+rm.reason+")","prune"));
              next={...next,customs:pruned};
            }
          }
          // ── Re-fit stale Gap/Lin algos ──
          if(curRows.length>=8&&curRows.length%4===0){
            let refitCount=0;
            next={...next,customs:(next.customs||[]).map(ca=>{
              if(!ca.generated||ca.benched)return ca;
              if(ca.name.startsWith("Gap_")){
                const col=ca.name.split("_")[1];
                const s=getSeries(col,curRows);if(s.length<4)return ca;
                const gs=[];for(let i=1;i<s.length;i++){let g=s[i]-s[i-1];if(g>50)g-=100;if(g<-50)g+=100;gs.push(g);}
                const ng=Math.round(M.mean(gs.slice(-6)));
                if(ng!==0&&Math.abs(ng-(parseInt(ca.name.split("_")[2])||0))>2){
                  refitCount++;
                  return{...ca,code:"(s,M)=>[M.mod(s[s.length-1]+("+ng+"))]",updatedAt:Date.now()};
                }
              }
              return ca;
            })};
            if(refitCount>0)syslog("♻ Re-fitted "+refitCount+" gap algo(s) to current drift","refit");
          }
          if(next!==cur)saveS(next);
          return next;
        });
      },100);
      return updated;
    });
    // Auto-advance date by 1 day and day# by 1
    if(dateIn){const nd=new Date(dateIn+"T12:00:00");nd.setDate(nd.getDate()+1);setDateIn(nd.getFullYear()+"-"+String(nd.getMonth()+1).padStart(2,"0")+"-"+String(nd.getDate()).padStart(2,"0"));}
    // Next day = max row in dataset + 1 (sequential, not 1-31 cycle)
    setS(cur=>{
      const curRows=cur.datasets&&cur.datasets[cur.active]?cur.datasets[cur.active].rows:[];
      const maxR=curRows.length?Math.max(...curRows.map(x=>x.row)):r;
      setRowIn(String(maxR+1));
      return cur;
    });
    setVals({A:"",B:"",C:"",D:""});
    st("Day "+pad2(r)+" saved ✓");
    syslog("📥 Day "+r+(dateIn?" ("+dateIn+")":"")+" saved","data");
  }

  function doBulk(){
    const lines=bulk.trim().split("\n").filter(l=>l.trim());let added=0,errs=0;
    const next=[...rows];
    lines.forEach(line=>{
      const pts=line.split(",").map(p=>p.trim());if(pts.length<5){errs++;return;}
      const row=parseInt(pts[0]);if(isNaN(row)||row<1||row>9999){errs++;return;}
      const abcd=pts.slice(1,5).map(p=>{if(!p||p.toUpperCase()==="XX")return null;const n=parseInt(p);return(isNaN(n)||n<0||n>99)?undefined:n;});
      if(abcd.some(v=>v===undefined)){errs++;return;}
      // 6th column optional: YYYY-MM-DD date
      const rowDate=pts[5]&&parseDate(pts[5])?pts[5]:undefined;
      const idx=next.findIndex(x=>x.row===row);
      const newEntry={row,A:abcd[0],B:abcd[1],C:abcd[2],D:abcd[3]};
      if(rowDate)newEntry.date=rowDate;
      if(idx>=0)next[idx]=newEntry;else next.push(newEntry);
      added++;
    });
    next.sort((a,b)=>a.row-b.row);setRows(()=>next);setBulk("");setShowBulk(false);
    // Auto-generate after bulk import
    setTimeout(()=>{
      setS(cur=>{
        const curRows=cur.datasets&&cur.datasets[cur.active]?cur.datasets[cur.active].rows:[];
        const _gc2=(cur.customs||[]).filter(a=>a.generated&&!a.benched).length;
        if(shouldAutoGenerate(curRows.length,cur.genN,cur.lastAutoGenRows,_gc2)){
          const existing=new Set([...Object.keys(A),...(cur.customs||[]).map(a=>a.name)]);
          const newAlgos=generateAlgos(curRows,existing);
          if(newAlgos.length>0){
            const msg="+"+newAlgos.length+" algos auto-generated after bulk import";
            const next2={...cur,customs:[...(cur.customs||[]),...newAlgos],genN:(cur.genN||0)+1,lastAutoGenRows:curRows.length};
            saveS(next2);
            setAutoGenLog(p=>[...p.slice(-4),msg]);
            return next2;
          }
        }
        return cur;
      });
    },150);
    st("Imported "+added+" rows"+(errs?", "+errs+" skipped":""),errs?"warn":"ok");
  }

  function doImport(e){
    const f=e.target.files[0];if(!f)return;
    const reader=new FileReader();
    reader.onload=ev=>{
      try{
        const parsed=JSON.parse(ev.target.result);
        if(parsed.dataset&&!parsed.datasets){parsed.datasets={def:{name:"Imported",rows:parsed.dataset}};parsed.active="def";}
        parsed.weights=migrateWeights(parsed.weights||{});
        if(!parsed.customs)parsed.customs=[];
        if(!parsed.patternBank)parsed.patternBank={A:{},B:{},C:{},D:{}};
        setS(parsed);saveS(parsed);
        st("Imported successfully");
      }catch(err){st("Import failed: "+err.message,"err");}
    };
    reader.readAsText(f);e.target.value="";
  }

  // ── CSV FILE IMPORT WITH AUTO-LEARN + PATTERN BANK UPDATE ──────────────
  function doImportCSV(e){
    const f=e.target.files[0];if(!f)return;
    const reader=new FileReader();
    reader.onload=ev=>{
      const text=ev.target.result;
      const lines=text.trim().split(/\r?\n/).filter(l=>l.trim()&&!l.toLowerCase().startsWith("row,a"));
      let added=0,updated=0,autoLearned=0,errs=0;
      const incomingRows=[];
      lines.forEach(line=>{
        const pts=line.split(",").map(p=>p.trim());
        if(pts.length<5){errs++;return;}
        const row=parseInt(pts[0]);
        if(isNaN(row)||row<1||row>9999){errs++;return;}
        const abcd=pts.slice(1,5).map(p=>{
          if(!p||p.toUpperCase()==="XX"||p==="?"||p==="(PRED)")return null;
          const n=parseInt(p);return(isNaN(n)||n<0||n>99)?undefined:n;
        });
        if(abcd.some(v=>v===undefined)){errs++;return;}
        const rowDate=pts[5]&&parseDate(pts[5])?pts[5]:undefined;
        incomingRows.push({row,A:abcd[0],B:abcd[1],C:abcd[2],D:abcd[3],date:rowDate});
      });
      if(!incomingRows.length){st("No valid rows in CSV","warn");e.target.value="";return;}

      upd(prev=>{
        const dsKey=prev.active;
        const existingRows=[...(prev.datasets[dsKey]?.rows||[])];
        let nw={...prev.weights};
        let nc={...prev.calibration||{A:{},B:{},C:{},D:{}}};
        let newCustoms=[...(prev.customs||[])];
        const newAccLog=[...(prev.accLog||[])];

        incomingRows.forEach(inc=>{
          const existIdx=existingRows.findIndex(r=>r.row===inc.row);
          const exist=existIdx>=0?existingRows[existIdx]:null;
          const newlyKnownCols=COLS.filter(c=>inc[c]!=null&&(exist?.[c]==null));
          if(exist===null){
            const entry={row:inc.row,A:inc.A,B:inc.B,C:inc.C,D:inc.D};
            if(inc.date)entry.date=inc.date;
            existingRows.push(entry);added++;
          } else {
            const merged={...exist};
            COLS.forEach(c=>{if(inc[c]!=null)merged[c]=inc[c];});
            if(inc.date&&!merged.date)merged.date=inc.date;
            existingRows[existIdx]=merged;
            if(newlyKnownCols.length>0)updated++;
          }
          // Auto-learn if this row was the predicted row
          if(newlyKnownCols.length>0&&prev.preds&&prev.predRow===inc.row){
            const actuals={};COLS.forEach(c=>{actuals[c]=inc[c]!=null?inc[c]:null;});
            const dateCtx=inc.date?parseDate(inc.date):null;
            const dtCtx=dateCtx?{dow:dateCtx.dow,lunar:dateCtx.lunarPhase,month:dateCtx.m,season:dateCtx.season}:null;
            newlyKnownCols.forEach(col=>{
              const colRegime=prev.preds[col]?prev.preds[col].regime:"normal";
              let uw=updateW(prev.preds[col],actuals[col],nw[col],inc.row,colRegime,nc[col]);
              uw.global=applyForgetting(uw.global,newAccLog);
              if(dtCtx)uw=updateDateWeights(prev.preds[col],actuals[col],uw,dtCtx);
              nw[col]=uw;
              if(prev.preds[col])nc[col]=updateCalibration(prev.preds[col].conf,actuals[col]===prev.preds[col].top5[0]?.value,nc[col]||{});
            });
            const exactCount=newlyKnownCols.filter(c=>prev.preds[c]&&prev.preds[c].top5[0]?.value===actuals[c]).length;
            newAccLog.push({at:new Date().toISOString(),targetRow:inc.row,date:inc.date||null,dateCtx:dtCtx,
              preds:Object.fromEntries(COLS.map(c=>[c,prev.preds[c]?.top5[0]?.value??null])),
              algoDetails:Object.fromEntries(COLS.map(c=>[c,prev.preds[c]?Object.fromEntries(Object.entries(prev.preds[c].details).sort((a,b)=>b[1].w-a[1].w).slice(0,20).map(([n,d])=>[n,d.pred])):{}])),
              actuals,exactCount,knownCount:newlyKnownCols.length,autoLearned:true});
            autoLearned++;
            syslog("🤖 Auto-learned row "+pad2(inc.row)+": "+exactCount+"/"+newlyKnownCols.length+" exact","learn");
          }
        });
        existingRows.sort((a,b)=>a.row-b.row);

        // Rebuild pattern bank with new data
        const newDs={...prev.datasets,[dsKey]:{...prev.datasets[dsKey],rows:existingRows}};
        const newPB={...prev.patternBank||{}};
        COLS.forEach(c=>{newPB[c]=updatePatternBank(prev.patternBank,c,newDs,newAccLog);});

        // Auto-generate algos
        const _gc=(newCustoms||[]).filter(a=>a.generated&&!a.benched).length;
        if(shouldAutoGenerate(existingRows.length,(prev.genN||0)+1,prev.lastAutoGenRows||0,_gc)){
          const ex2=new Set([...Object.keys(A),...newCustoms.map(a=>a.name)]);
          const fa=generateAlgos(existingRows,ex2);
          if(fa.length>0){newCustoms=[...newCustoms,...fa];syslog("🔧 +"+fa.length+" algos auto-generated after CSV import","gen");}
        }
        const parts=[];
        if(added)parts.push(added+" new");if(updated)parts.push(updated+" updated");
        if(autoLearned)parts.push("🤖 "+autoLearned+" auto-learned");if(errs)parts.push(errs+" skipped");
        syslog("📂 CSV import: "+parts.join(", "),"info");

        return{...prev,weights:nw,calibration:nc,customs:newCustoms,datasets:newDs,patternBank:newPB,
          accLog:newAccLog.slice(-365),genN:(prev.genN||0)+1,lastAutoGenRows:existingRows.length};
      });
      const parts=[];
      if(added)parts.push(added+" rows added");if(updated)parts.push(updated+" updated");
      if(autoLearned)parts.push("🤖 "+autoLearned+" auto-learned");if(errs)parts.push(errs+" skipped");
      st("CSV: "+parts.join(", "),errs&&!added&&!updated?"warn":"ok");
    };
    reader.readAsText(f);e.target.value="";
  }

  function doImportWeights(e){
    const f=e.target.files[0];if(!f)return;
    const reader=new FileReader();
    reader.onload=ev=>{
      try{
        const parsed=JSON.parse(ev.target.result);
        const data=parseImportedWeights(parsed);
        if(!data){setWeightsMsg("❌ Invalid weights file");return;}
        if(!data.weights||!data.weights.A||!data.weights.A.global){
          data.weights={A:{global:{},perRow:{},perRange:{}},B:{global:{},perRow:{},perRange:{}},C:{global:{},perRow:{},perRange:{}},D:{global:{},perRow:{},perRange:{}}};
        }
        upd(prev=>({...prev,weights:data.weights,customs:data.customs||prev.customs,accLog:data.accLog||prev.accLog}));
        const sessCount=data.accLog?data.accLog.length:0;
        const algoCount=data.customs?data.customs.length:0;
        setWeightsMsg("✅ Weights loaded — "+sessCount+" sessions, "+algoCount+" algos");
        st("Weights imported ✓");
      }catch(err){setWeightsMsg("❌ Parse error: "+err.message);}
    };
    reader.readAsText(f);e.target.value="";
  }

  function runPredict(){
    if(rows.length<4){st("Need at least 4 rows","warn");return;}
    st("Computing…","busy");
    const maxRow=Math.max(...rows.map(r=>r.row));
    const target=maxRow+1; // sequential global day — no monthly cycling
    // Compute target date from the latest dated row in dataset.
    // Keep manual override only when it is beyond the latest dataset date.
    const latestDated=rows.reduce((mx,r)=>{
      if(!r?.date||!parseDate(r.date))return mx;
      return!mx||r.date>mx?r.date:mx;
    },"");
    const autoDate=latestDated?nextDateISO(latestDated):"";
    const tDate=(predDate&&(!latestDated||predDate>latestDated))?predDate:autoDate;
    const result={};
    const _sharedMeta=(S.accLog||[]).length>=3?buildMetaModel(S.accLog):null;
    const _slimAccLog=(S.accLog||[]).slice(-10);
    COLS.forEach(col=>{
      const W={...S.weights[col],_metaModel:_sharedMeta,_accLog:_slimAccLog,_leaderboard:(S.algoLeaderboard&&S.algoLeaderboard[col])||{}};
      result[col]=predictCol(col,rows,W,S.customs,tDate,S.datasets,S.patternBank);
    });
    // Joint column hint
    const knownPreds={};
    COLS.forEach(col=>{if(result[col])knownPreds[col]=result[col].top5[0]?.value;});
    // ── ROW SUM TARGET: inject after all 4 cols predicted ──
    // NOTE: jointColHint intentionally removed from runPredict — it propagated
    // prediction errors (if col A prediction is wrong, it corrupts col B hint).
    // RowSumTarget is kept because it uses historical sum distribution, not predictions.
    COLS.forEach(col=>{
      if(!result[col])return;
      const sumSigs=getRowSumSignal(col,rows,knownPreds);
      Object.entries(sumSigs).forEach(([name,preds])=>{
        if(!preds||!preds.length)return;
        const top=result[col].top5;
        const v=preds[0];
        const existing=top.find(p=>p.value===v);
        if(existing)existing.pct=Math.min(99,existing.pct+12);
        else if(top.length>0)top.push({value:v,votes:1,pct:12,algos:[name]});
      });
    });
    // ── MULTI-STEP SUM CONSENSUS ───────────────────
    // If predicted A+B+C+D deviates >15% from historical sum mean, nudge all toward target
    const complete=rows.filter(r=>COLS.every(c=>ok(r[c])));
    if(complete.length>=4){
      const histSums=complete.map(r=>r.A+r.B+r.C+r.D);
      const sumMean=M.mean(histSums),sumStd=M.std(histSums);
      const predSum=COLS.reduce((s,c)=>s+(result[c]?.top5[0]?.value||0),0);
      const deviation=predSum-sumMean;
      if(Math.abs(deviation)>sumStd*1.5&&sumStd>0){
        const correction=Math.round(deviation/4*0.4); // distribute 40% correction across 4 cols
        COLS.forEach(col=>{
          if(!result[col]?.top5[0])return;
          const nudged=M.mod(result[col].top5[0].value-correction);
          // Check nudged value is historically plausible
          const colSeries=rows.map(r=>r[col]).filter(ok);
          const colMean=M.mean(colSeries),colStd=M.std(colSeries);
          if(Math.abs(nudged-colMean)<=colStd*2){
            // Inject nudged as a top candidate
            const existing=result[col].top5.find(p=>p.value===nudged);
            if(existing)existing.pct=Math.min(99,existing.pct+10);
            else result[col].top5.push({value:nudged,votes:0,pct:10,algos:["SumConsensus"]});
          }
        });
      }
    }
    upd(prev=>({...prev,preds:result,predRow:target,predDate:tDate}));
    setPredDate(tDate||"");
    setCheckRes(null);setActs({A:"",B:"",C:"",D:""});setTab("predict");
    setTimeout(()=>st("Predictions ready"+(tDate?" · "+tDate:"")+" ✓"),300);
  }

  // Run full prediction for unknown columns using known values as cross-col hints
  function predictMissingCols(){
    const known={};
    let hasKnown=false;
    COLS.forEach(col=>{
      const raw=acts[col].trim();
      if(raw&&raw.toUpperCase()!=="XX"&&!isNaN(parseInt(raw))&&parseInt(raw)>=0&&parseInt(raw)<=99){
        known[col]=parseInt(raw);
        hasKnown=true;
      }
    });
    if(!hasKnown){st("Enter at least one known column value first","warn");return;}
    const missing=COLS.filter(col=>known[col]==null);
    if(!missing.length){st("All columns already filled","warn");return;}
    st("Predicting missing columns…","busy");

    // Build a temporary dataset that includes a partial row with known values
    // so cross-col algorithms can use known values as hints
    const tempRow={row:S.predRow||0};
    COLS.forEach(col=>{
      tempRow[col]=known[col]!=null?known[col]:null;
    });
    const tempDataset=[...rows,tempRow];

    const newActs={...acts};
    const partialPreds={};

    // Build meta model once, outside the loop
    const _meta2=(S.accLog||[]).length>=3?buildMetaModel(S.accLog):null;
    missing.forEach(col=>{
      const pred=predictCol(col,tempDataset,{...S.weights[col],_metaModel:_meta2,_leaderboard:(S.algoLeaderboard&&S.algoLeaderboard[col])||{}},S.customs,S.predDate||"",S.datasets,S.patternBank);
      if(pred&&pred.top5[0]){
        const top=pred.top5[0].value;
        newActs[col]=String(top).padStart(2,"0");
        partialPreds[col]=pred;
      }
    });

    setActs(newActs);
    // Store partial predictions so user can see top-5 for each missing col
    setMissingPreds(partialPreds);
    const knownStr=Object.entries(known).map(([c,v])=>c+"="+pad2(v)).join(", ");
    setTimeout(()=>st("Predicted "+missing.join(",")+" using "+knownStr+" as cross-col hint ✓"),300);
  }

  function checkAndLearn(){
    if(!S.preds){st("Run prediction first","warn");return;}
    const actuals={};
    const knownCols=[];
    for(let i=0;i<COLS.length;i++){
      const col=COLS[i],raw=acts[col].trim();
      // Allow XX or empty = unknown column
      if(!raw||raw.toUpperCase()==="XX"){actuals[col]=null;continue;}
      const n=parseInt(raw);if(isNaN(n)||n<0||n>99){st(col+" must be 00–99 or XX","warn");return;}
      actuals[col]=n;
      knownCols.push(col);
    }
    // Need at least 1 known column to learn
    if(knownCols.length===0){st("Enter at least one actual value (use XX for unknown)","warn");return;}
    const results={};let exactCount=0;
    COLS.forEach(col=>{
      if(actuals[col]===null){
        // Unknown: mark as skipped, use top prediction as placeholder
        const top1=S.preds[col]&&S.preds[col].top5[0]?S.preds[col].top5[0].value:null;
        results[col]={predicted:top1,actual:null,exact:false,near:false,skipped:true};
        return;
      }
      const top1=S.preds[col]&&S.preds[col].top5[0]?S.preds[col].top5[0].value:null;
      const actual=actuals[col];
      const ex=top1===actual,nr=!ex&&M.near(top1!=null?top1:-1,actual,2);
      results[col]={predicted:top1,actual,exact:ex,near:nr,skipped:false};
      if(ex)exactCount++;
    });
    setCheckRes(results);
    upd(prev=>{
      const nw={...prev.weights};
      const nc={...prev.calibration||{A:{},B:{},C:{},D:{}}};
      // Bug 2 fix: compute dateCtx BEFORE the loop that uses it
      const _datePd=prev.predDate?parseDate(prev.predDate):null;
      const dateCtx=_datePd?{dow:_datePd.dow,lunar:_datePd.lunarPhase,month:_datePd.m,season:_datePd.season}:null;
      COLS.forEach(col=>{
        if(actuals[col]===null)return;
        const colRegime=prev.preds[col]?prev.preds[col].regime:"normal";
        let updated=updateW(prev.preds[col],actuals[col],prev.weights[col],prev.predRow,colRegime,prev.calibration&&prev.calibration[col]);
        // Bug 12 fix: decay ALL weight maps, not just global
        updated.global=applyForgetting(updated.global,prev.accLog);
        // Soft decay perRow/perRange/perRegime toward 1.0 (prevent stale boosts)
        const decayMap=(map)=>{
          const d={};
          Object.entries(map||{}).forEach(([k,v])=>{
            if(typeof v==='object'&&v!==null){
              d[k]={};
              Object.entries(v).forEach(([n,w])=>{
                if(typeof w==='number')d[k][n]=+(w>1?w*0.98+(1-0.98):w*0.99+(1-0.99)).toFixed(4);
                else d[k][n]=w;
              });
            }
          });
          return d;
        };
        updated.perRow=decayMap(updated.perRow);
        updated.perRange=decayMap(updated.perRange);
        updated.perRegime=decayMap(updated.perRegime);
        updated.perDOW=decayMap(updated.perDOW||{});
        updated.perLunar=decayMap(updated.perLunar||{});
        // Fix 8: apply date-conditioned weight update
        if(dateCtx)updated=updateDateWeights(prev.preds[col],actuals[col],updated,dateCtx);
        nw[col]=updated;
        const pred=prev.preds[col];
        if(pred)nc[col]=updateCalibration(pred.conf,actuals[col]===pred.top5[0]?.value,nc[col]||{});
      });
      const knownCount=knownCols.length;
      // Fix 3: store full top5 AND per-algo predictions for meta-learner
      const predsTop1=Object.fromEntries(COLS.map(c=>[c,prev.preds[c]&&prev.preds[c].top5[0]?prev.preds[c].top5[0].value:null]));
      // Bug 10 fix: store only top-20 algos by weight to prevent localStorage overflow
      // (100+ algos × 4 cols × 99 sessions would exceed 5MB limit)
      const algoDetails=Object.fromEntries(COLS.map(c=>[c,
        prev.preds[c]?Object.fromEntries(
          Object.entries(prev.preds[c].details)
            .sort((a,b)=>b[1].w-a[1].w)
            .slice(0,20)
            .map(([n,d])=>[n,d.pred])
        ):{}
      ]));
      const entry={at:new Date().toISOString(),targetRow:prev.predRow,date:prev.predDate||null,dateCtx,
        preds:predsTop1,algoDetails,actuals,results,exactCount,knownCount};
      // Auto-prune weak generated algos
      const curRows=prev.datasets&&prev.datasets[prev.active]?prev.datasets[prev.active].rows:[];
      const {pruned,removed}=pruneWeakAlgos(prev.customs||[],nw,curRows);
      if(removed.length>0){
        const msgs=removed.map(r=>"Pruned: "+r.name+" ("+r.reason+")");
        setAutoGenLog(p=>[...p.slice(-(5-msgs.length)),...msgs]);
      }

      // Auto-add actual result as new dataset row (null for unknown cols)
      const autoRow={row:prev.predRow};
      COLS.forEach(col=>{autoRow[col]=actuals[col]!=null?actuals[col]:null;});
      // Preserve date on the learned row so future date-algo training includes it
      if(prev.predDate)autoRow.date=prev.predDate;
      const dsKey=prev.active;
      const existingRows=prev.datasets[dsKey]?prev.datasets[dsKey].rows:[];
      const newRows=[...existingRows.filter(r=>r.row!==prev.predRow),autoRow].sort((a,b)=>a.row-b.row);
      const newDs={...prev.datasets,[dsKey]:{...prev.datasets[dsKey],rows:newRows}};
      // ── Post-learn auto-generate: new row means new patterns may be detectable ──
      let finalCustoms=pruned;
      const _gc3=finalCustoms.filter(a=>a.generated&&!a.benched).length;
      if(shouldAutoGenerate(newRows.length,(prev.genN||0)+1,prev.lastAutoGenRows||0,_gc3)){
        const existing2=new Set([...Object.keys(A),...finalCustoms.map(a=>a.name)]);
        const freshAlgos=generateAlgos(newRows,existing2);
        if(freshAlgos.length>0)finalCustoms=[...finalCustoms,...freshAlgos];
      }

      // ── Rebuild Pattern Bank: update after every learn with full dataset context ──
      // This keeps the cross-period long-term memory current.
      // Every COLS entry gets updated with ALL datasets so 2026 predictions can use 2025 patterns.
      const newPatternBank={...prev.patternBank||{}};
      const newFullLog=[...(prev.accLog||[]).slice(-365),entry];
      COLS.forEach(c=>{
        if(actuals[c]!=null){ // only rebuild for cols where we just learned
          newPatternBank[c]=updatePatternBank(prev.patternBank,c,newDs,newFullLog);
        }
      });

      return{...prev,weights:nw,calibration:nc,customs:finalCustoms,datasets:newDs,patternBank:newPatternBank,accLog:newFullLog,lastAutoGenRows:newRows.length};
    });
    st("Learned! "+exactCount+"/4 exact — weights saved ✓");
    const pct=Math.round(exactCount/knownCols.length*100);
    syslog((exactCount===4?"🎯":"📊")+" Learn result: "+exactCount+"/"+knownCols.length+" exact ("+pct+"%)","learn");
    if(exactCount===4)syslog("✨ Perfect prediction! Pattern well-captured — consider exporting weights.","alert");
  }

  // ════════════════════════════════════════════════════════════════════════
  // ── BULK AUTO-TRAIN ─────────────────────────────────────────────────────
  // Walk-forward simulation on historical data with known outcomes.
  //
  // Flow per row i:
  //   1. Use rows 0..i-1 as training history
  //   2. Predict row i using predictCol (all algos, cross-col, date signals)
  //   3. Compare prediction vs actual (row i)
  //   4. Update weights exactly as checkAndLearn does
  //   5. Prune weak algos, generate new ones, update pattern bank
  //
  // Result: weights trained on ALL historical rows in correct temporal order.
  // Best algos surface, worst algos get pruned. Pattern bank fully populated.
  // ════════════════════════════════════════════════════════════════════════

  function parseAutoTrainCSV(text){
    // Accepts: Row,A,B,C,D,Date  OR  Row,A,B,C,D  (date optional)
    // First line may be a header — auto-detected and skipped
    const lines=text.trim().split(/\r?\n/).filter(l=>l.trim());
    const rows=[];
    let skipped=0;
    lines.forEach((line,idx)=>{
      // Skip header
      if(idx===0&&/[a-zA-Z]{2,}/.test(line.split(",")[0]))return;
      const pts=line.split(",").map(p=>p.trim());
      if(pts.length<5){skipped++;return;}
      const rowNum=parseInt(pts[0]);
      if(isNaN(rowNum)||rowNum<1||rowNum>99999){skipped++;return;}
      const abcd=pts.slice(1,5).map(p=>{
        if(!p||p.toUpperCase()==="XX"||p==="?"||p==="(PRED)")return null;
        const n=parseInt(p);
        return(isNaN(n)||n<0||n>99)?undefined:n;
      });
      if(abcd.some(v=>v===undefined)){skipped++;return;}
      // At least one column must be known
      if(abcd.every(v=>v===null)){skipped++;return;}
      const rowDate=pts[5]&&parseDate(pts[5])?pts[5]:null;
      rows.push({row:rowNum,A:abcd[0],B:abcd[1],C:abcd[2],D:abcd[3],date:rowDate});
    });
    return{rows,skipped};
  }

  function doAutoTrain(e){
    const f=e.target.files[0];if(!f)return;
    e.target.value="";
    const reader=new FileReader();
    reader.onload=ev=>{
      const{rows:csvRows,skipped}=parseAutoTrainCSV(ev.target.result);
      if(csvRows.length<4){
        st("Need at least 4 rows with known outcomes in CSV","warn");
        return;
      }
      // Sort by date then row number to ensure correct temporal order
      csvRows.sort((a,b)=>{
        if(a.date&&b.date)return a.date.localeCompare(b.date);
        return a.row-b.row;
      });

      autoTrainRef.current=false;
      setAutoTrainStatus({running:true,progress:0,total:csvRows.length,done:false,log:[
        "📂 Loaded "+csvRows.length+" rows"+(skipped?" ("+skipped+" skipped)":""),
        "⚙ Starting walk-forward training simulation…"
      ],result:null});

      // Run async in chunks to keep UI responsive
      setTimeout(()=>runAutoTrainLoop(csvRows),50);
    };
    reader.readAsText(f);
  }

  function runAutoTrainLoop(csvRows){
    // ── Snapshot state into a plain JS object held in autoTrainRef ──────
    // Nothing inside this object touches React state during the loop.
    // One final setS at the end commits everything atomically.
    setS(prev=>{
      _TC.clear(); // clear module caches — fresh training session
      const dsKey=prev.active;
      autoTrainRef.current={
        csvRows,index:0,total:csvRows.length,dsKey,
        historyRows:[...(prev.datasets[dsKey]?.rows||[])],
        weights:JSON.parse(JSON.stringify(prev.weights)),
        calibration:JSON.parse(JSON.stringify(prev.calibration||{A:{},B:{},C:{},D:{}})),
        customs:JSON.parse(JSON.stringify(prev.customs||[])),
        accLog:[...(prev.accLog||[])],
        patternBank:JSON.parse(JSON.stringify(prev.patternBank||{A:{},B:{},C:{},D:{}})),
        allDs:{...prev.datasets}, // kept in sync as rows are added
        exactTotal:0,nearTotal:0,totalKnown:0,
        btCache:{},         // {col__name:{bt,wfBoost}} — rebuilt on schedule
        btCacheAge:0,       // ticks since last btCache refresh
        cachedMeta:null,    // buildMetaModel result — rebuilt every 25 rows
        metaModelAge:0,
        logLines:[],        // milestone lines accumulated during run
        perf:{rows:0,predMs:0,btMs:0,updMs:0,insertMs:0,tickMs:0,startedAt:PERF_NOW()},
        lightweight:false,
        heavyStreak:0,
        leaderboard:{A:{},B:{},C:{},D:{}}
      };
      setTimeout(autoTrainTick,0);
      return prev; // no state change yet
    });
  }

  // ── Single tick: processes BATCH_SIZE rows then yields to browser ──────
  function autoTrainTick(){
    const tickStart=PERF_NOW();
    const tr=autoTrainRef.current;
    if(!tr||tr===false){
      setAutoTrainStatus(s=>s?{...s,running:false,done:true,log:[...(s?.log||[]),"⛔ Cancelled"]}:s);
      return;
    }
    if(tr.index>=tr.total){autoTrainFinish(tr);return;}

    // Adaptive chunk with hard frame budget so browser stays responsive
    const hlen=tr.historyRows.length;
    const baseRows=tr.lightweight?(hlen<60?2:4):(hlen<15?1:hlen<50?3:hlen<200?6:10);
    const maxRows=Math.max(1,baseRows);
    const frameBudgetMs=tr.lightweight?7.5:11.5;
    let processed=0;
    while(tr.index<tr.total&&processed<maxRows){
      if(PERF_NOW()-tickStart>frameBudgetMs)break;
      const bi=tr.index;
      if(autoTrainRef.current===false)break;
      const csvRow=tr.csvRows[bi];
      const knownCols=COLS.filter(c=>csvRow[c]!=null);

      // Rows with no values — add to history only
      if(!knownCols.length){
        const insT0=PERF_NOW();
        _atInsert(tr,csvRow);
        tr.perf.insertMs+=PERF_NOW()-insT0;
        tr.index++;
        processed++;
        tr.perf.rows++;
        continue;
      }

      // ── btCache refresh schedule ──────────────────────────────────────
      // First run always. Then: every 5 rows early, every 20 mid, every 40 late.
      // walkFwd only starts after enough history to be meaningful (>=20 rows).
      // This is the single biggest perf lever — don't refresh too often.
      const cacheInterval=tr.lightweight?(hlen<60?18:40):(hlen<15?5:hlen<60?15:hlen<200?30:50);
      if(!tr.btCacheAge||tr.btCacheAge>=cacheInterval){
        const btT0=PERF_NOW();
        _atRefreshBtCache(tr);
        tr.perf.btMs+=PERF_NOW()-btT0;
        tr.btCacheAge=1;
      } else {tr.btCacheAge++;}

      // ── metaModel refresh every 30 rows ──────────────────────────────
      if(tr.accLog.length>=3&&(!tr.metaModelAge||tr.metaModelAge>=30)){
        tr.cachedMeta=buildMetaModel(tr.accLog);
        tr.metaModelAge=1;
      } else if(tr.metaModelAge){tr.metaModelAge++;}

      // ── Predict ──────────────────────────────────────────────────────
      const rowPreds={};
      const predT0=PERF_NOW();
      knownCols.forEach(col=>{
        const W={...tr.weights[col],
          _metaModel:tr.cachedMeta,
          _accLog:tr.accLog.slice(-6),   // only last 6 for dead-zone correction
          _btCache:tr.btCache,_btCacheCol:col,
          _leaderboard:tr.leaderboard[col]||{},
          _algoBudget:tr.lightweight?10:undefined,
          _lightweight:tr.lightweight};
        rowPreds[col]=predictCol(col,tr.historyRows,W,tr.customs,csvRow.date||"",tr.allDs,tr.patternBank);
      });
      const predMs=PERF_NOW()-predT0;
      tr.perf.predMs+=predMs;
      if(predMs>HEAVY_PRED_THRESHOLD_MS)tr.heavyStreak=Math.min(MAX_HEAVY_STREAK,(tr.heavyStreak||0)+1);
      else tr.heavyStreak=Math.max(0,(tr.heavyStreak||0)-1);
      tr.lightweight=tr.heavyStreak>=LIGHTWEIGHT_TRIGGER_STREAK||tr.historyRows.length>LIGHTWEIGHT_HISTORY_THRESHOLD;

      // ── Compare & weight update ───────────────────────────────────────
      const dateCtxPd=csvRow.date?parseDate(csvRow.date):null;
      const dateCtx=dateCtxPd?{dow:dateCtxPd.dow,lunar:dateCtxPd.lunarPhase,month:dateCtxPd.m,season:dateCtxPd.season}:null;
      let rowExact=0,rowKnown=0;
      const rowResults={};
      const updT0=PERF_NOW();
      knownCols.forEach(col=>{
        const actual=csvRow[col];
        const pred=rowPreds[col];
        if(!pred){rowResults[col]={predicted:null,actual,exact:false,near:false,skipped:true};return;}
        const top1=pred.top5[0]?.value;
        const ex=top1===actual,nr=!ex&&M.near(top1??-1,actual,2);
        rowResults[col]={predicted:top1,actual,exact:ex,near:nr,skipped:false};
        if(ex)rowExact++;rowKnown++;
        const regime=pred.regime||"normal";
        // Skip applyForgetting in hot path — deferred to autoTrainFinish
        let uw=updateW(pred,actual,tr.weights[col],csvRow.row,regime,tr.calibration[col]);
        if(dateCtx)uw=updateDateWeights(pred,actual,uw,dateCtx);
        tr.weights[col]=uw;
        tr.calibration[col]=updateCalibration(pred.conf,ex,tr.calibration[col]||{});
        tr.leaderboard=updateAlgoLeaderboard(tr.leaderboard,col,pred);
      });
      tr.perf.updMs+=PERF_NOW()-updT0;
      tr.exactTotal+=rowExact;
      tr.nearTotal+=Object.values(rowResults).filter(r=>r.near).length;
      tr.totalKnown+=rowKnown;

      // ── Insert row into sorted history (binary insert — O(log n), not O(n log n)) ──
      const insT0=PERF_NOW();
      _atInsert(tr,csvRow);
      tr.perf.insertMs+=PERF_NOW()-insT0;

      // ── Compact accLog entry ──────────────────────────────────────────
      tr.accLog.push({
        at:bi,targetRow:csvRow.row,date:csvRow.date||null,dateCtx,
        preds:Object.fromEntries(COLS.map(c=>[c,rowPreds[c]?.top5[0]?.value??null])),
        algoDetails:Object.fromEntries(COLS.map(c=>[c,rowPreds[c]
          ?Object.fromEntries(Object.entries(rowPreds[c].details||{}).sort((a,b)=>b[1].w-a[1].w).slice(0,15).map(([n,d])=>[n,d.pred]))
          :{}])),
        actuals:Object.fromEntries(COLS.map(c=>[c,csvRow[c]??null])),
        results:rowResults,exactCount:rowExact,knownCount:rowKnown,autoTrained:true
      });
      if(tr.accLog.length>365)tr.accLog=tr.accLog.slice(-365);

      // ── Generate algos every 8 rows ───────────────────────────────────
      if(tr.historyRows.length>=8&&bi>0&&bi%8===0){
        const genCount=tr.customs.filter(a=>a.generated&&!a.benched).length;
        if(genCount<MAX_GENERATED_ALGOS){
          const existing=new Set([...Object.keys(A),...tr.customs.map(a=>a.name)]);
          const fresh=generateAlgos(tr.historyRows,existing);
          if(fresh.length)tr.customs=[...tr.customs,...fresh];
        }
      }

      // ── Prune every 40 rows, but ONLY after 50+ history rows ─────────
      // Too-early pruning eliminates algos before they've had enough data
      // to demonstrate their value — causes the "1 algo" problem.
      if(bi>0&&bi%40===0&&tr.historyRows.length>=50){
        const{pruned}=pruneWeakAlgos(tr.customs,tr.weights,tr.historyRows,tr.btCache);
        tr.customs=pruned;
      }

      // ── Milestone log ─────────────────────────────────────────────────
      const milestones=[
        Math.floor(tr.total*0.10),Math.floor(tr.total*0.25),
        Math.floor(tr.total*0.50),Math.floor(tr.total*0.75),tr.total-1
      ];
      if(milestones.includes(bi)){
        const pct=tr.totalKnown>0?Math.round(tr.exactTotal/tr.totalKnown*100):0;
        tr.logLines.push("📊 "+(bi+1)+"/"+tr.total+" — "+tr.exactTotal+"/"+tr.totalKnown+" exact ("+pct+"%)");
      }
      tr.index++;
      processed++;
      tr.perf.rows++;
    }
    tr.perf.tickMs+=PERF_NOW()-tickStart;

    // UI update: only every 10 rows or on milestone
    if(tr.logLines.length||tr.index%10===0||tr.index>=tr.total){
      const lines=[...tr.logLines];tr.logLines=[];
      setAutoTrainStatus(s=>({...s,progress:tr.index,
        log:lines.length?[...(s?.log||[]),...lines]:s?.log||[]}));
    }
    setTimeout(autoTrainTick,0);
  }

  // ── Binary-insert into sorted historyRows ─────────────────────────────
  // Dedup key = date+row so cycling row numbers (1-31 monthly) all stay in history
  function _atInsert(tr,row){
    const h=tr.historyRows;
    // Match on BOTH row number AND date (if present) — prevents month-cycle overwrites
    const deupKey=row.date?row.date+"_"+row.row:null;
    const ex=h.findIndex(r=>{
      if(deupKey&&r.date)return r.date===row.date&&r.row===row.row; // exact date match
      if(!row.date&&!r.date)return r.row===row.row; // no dates: fallback to row# only
      return false; // don't overwrite dated row with undated or vice versa
    });
    if(ex>=0)h.splice(ex,1);
    // Sort by date first, then row number
    let lo=0,hi=h.length;
    while(lo<hi){
      const m=(lo+hi)>>1;
      const hm=h[m];
      const cmp=row.date&&hm.date?row.date.localeCompare(hm.date):hm.row<=row.row?0:1;
      cmp<=0?lo=m+1:hi=m;
    }
    h.splice(lo,0,row);
    tr.allDs[tr.dsKey]={rows:h};
    _TC.bumpVer();
  }

  // ── Rebuild btScore+walkFwd cache for all algos ───────────────────────
  function _atRefreshBtCache(tr){
    const cache={};
    const hlen=tr.historyRows.length;
    // walkFwd only when history is long enough to be reliable
    const doWalkFwd=hlen>=25;
    COLS.forEach(col=>{
      const localSer=getSeries(col,tr.historyRows);
      const globalSer=getGlobalSeries(col,tr.allDs);
      // Cap at 120 hard — beyond this btScore gain is <1% but cost grows linearly
      const btSer=(globalSer.length>localSer.length?globalSer:localSer).slice(-120);
      if(btSer.length<4)return;
      const regime=getRegime(localSer);
      const allowedNames=ALGO_NAMES.filter(n=>algoAllowed(n,regime));
      let budget=32;
      if(tr.lightweight)budget=10;
      else if(hlen>280)budget=16;
      else if(hlen>160)budget=22;
      const evalNames=selectAlgoNames(allowedNames,regime,Math.min(budget,allowedNames.length));
      evalNames.forEach(name=>{
        try{
          const bt=btScore(A[name],btSer);
          let wfBoost=1.0;
          if(doWalkFwd&&btSer.length>=15){
            const wf=walkFwd(A[name],btSer);
            if(wf&&wf.total>=3){const r=(wf.exact+wf.near*0.4)/wf.total;wfBoost=0.6+r*0.8;}
          }
          cache[col+"__"+name]={bt,wfBoost};
        }catch(e){cache[col+"__"+name]={bt:0.05,wfBoost:1.0};}
      });
      // Custom algos
      const customBudget=tr.lightweight?8:14;
      tr.customs.filter(ca=>ca.enabled&&ca.code).slice(0,customBudget).forEach(ca=>{
        try{
          const fn=makeCustomFn(ca.code);if(!fn)return;
          const bt=btScore(fn,btSer);
          let wfBoost=1.0;
          if(doWalkFwd&&btSer.length>=15){
            const wf=walkFwd(fn,btSer);
            if(wf&&wf.total>=3){const r=(wf.exact+wf.near*0.4)/wf.total;wfBoost=0.6+r*0.8;}
          }
          cache[col+"__"+ca.name]={bt,wfBoost,fn};
        }catch(e){}
      });
    });
    tr.btCache=cache;
  }

  function autoTrainFinish(tr){
    // Apply forgetting once at the end instead of every row
    COLS.forEach(col=>{
      if(tr.weights[col]?.global)
        tr.weights[col].global=applyForgetting(tr.weights[col].global,tr.accLog);
    });
    // Final prune + pattern bank rebuild
    const{pruned:finalCustoms,removed}=pruneWeakAlgos(tr.customs,tr.weights,tr.historyRows,tr.btCache);
    const finalDs={...tr.allDs,[tr.dsKey]:{rows:tr.historyRows}};
    COLS.forEach(c=>{tr.patternBank[c]=updatePatternBank(tr.patternBank,c,finalDs,tr.accLog);});

    const exactPct=tr.totalKnown>0?Math.round(tr.exactTotal/tr.totalKnown*100):0;
    const nearPct=tr.totalKnown>0?Math.round((tr.exactTotal+tr.nearTotal)/tr.totalKnown*100):0;
    const runtimeSec=Math.max(0.1,(PERF_NOW()-tr.perf.startedAt)/1000);
    const rowsPerSec=+((tr.perf.rows||tr.total)/runtimeSec).toFixed(2);
    const avgPredMs=tr.perf.rows?+(tr.perf.predMs/tr.perf.rows).toFixed(2):0;
    const avgTickMs=tr.perf.rows?+(tr.perf.tickMs/tr.perf.rows).toFixed(2):0;
    const finalLog=[
      "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
      "✅ Done — "+tr.total+" rows processed",
      "🎯 Exact: "+tr.exactTotal+"/"+tr.totalKnown+" ("+exactPct+"%)  Near+: "+nearPct+"%",
      "⏱ Throughput: "+rowsPerSec+" rows/s · avg predict "+avgPredMs+"ms · avg tick "+avgTickMs+"ms",
      "🧠 Algos kept: "+finalCustoms.filter(a=>!a.benched).length+"  Pruned: "+removed.length,
      "🗂 Pattern bank: "+COLS.reduce((s,c)=>s+Object.keys(tr.patternBank[c]?.mdProfiles||{}).length,0)+" MD-exact profiles",
      "💾 Weights saved — ready to predict"
    ];

    // Single atomic React state commit
    setS(prev=>{
      const saved={...prev,
        weights:tr.weights,calibration:tr.calibration,
        customs:finalCustoms,datasets:finalDs,
        patternBank:tr.patternBank,
        accLog:tr.accLog.slice(-365),
        algoLeaderboard:tr.leaderboard,
        genN:(prev.genN||0)+1,lastAutoGenRows:tr.historyRows.length
      };
      saveS(saved);
      return saved;
    });
    setAutoTrainStatus(s=>({...s,running:false,done:true,progress:tr.total,
      log:[...(s?.log||[]).filter(l=>!l.startsWith("━")),...finalLog],
      result:{exactPct,nearPct,total:tr.total,
        kept:finalCustoms.filter(a=>!a.benched).length,pruned:removed.length,
        perf:{rowsPerSec,avgPredMs,avgTickMs}}}));
    autoTrainRef.current=false;
    _TC.clear(); // free cache memory
    syslog("🚀 Auto-train done: "+tr.total+" rows · "+exactPct+"% exact · "+finalCustoms.filter(a=>!a.benched).length+" algos","learn");
  }

  function doGenerate(){
    if(rows.length<6){st("Need at least 6 rows","warn");return;}
    const existing=new Set([...Object.keys(A),...(S.customs||[]).map(a=>a.name)]);
    const newAlgos=generateAlgos(rows,existing);
    if(!newAlgos.length){setGenMsg("No new patterns found — add more rows");return;}
    upd(prev=>({...prev,customs:[...(prev.customs||[]),...newAlgos],genN:(prev.genN||0)+1}));
    setGenMsg("✓ "+newAlgos.length+" new adaptive algorithms generated & saved");
    st("Generated "+newAlgos.length+" algos ✓");
  }

  function doRebuildPatternBank(){
    if(rows.length<4){st("Need at least 4 rows with dates","warn");return;}
    upd(prev=>{
      const newPB={...prev.patternBank||{}};
      COLS.forEach(c=>{
        newPB[c]=updatePatternBank(prev.patternBank,c,prev.datasets,prev.accLog||[]);
      });
      const mdCount=COLS.reduce((s,c)=>s+Object.keys(newPB[c]?.mdProfiles||{}).length,0);
      const dyCount=COLS.reduce((s,c)=>s+Object.keys(newPB[c]?.dayProfiles||{}).length,0);
      const moCount=COLS.reduce((s,c)=>s+Object.keys(newPB[c]?.monthlyProfiles||{}).length,0);
      syslog("🧠 Pattern bank rebuilt: "+moCount+" monthly + "+dyCount+" day-profiles + "+mdCount+" MD-exact profiles across 4 cols","info");
      st("Pattern bank rebuilt ✓ ("+mdCount+" patterns)");
      return{...prev,patternBank:newPB};
    });
  }

  function doTournament(){
    if(!S.customs||S.customs.length<4){st("Need at least 4 custom algos","warn");return;}
    const evolved=runTournament(S.customs,rows,S.weights);
    upd(prev=>({...prev,customs:evolved,tourN:(prev.tourN||0)+1}));
    st("Tournament #"+((S.tourN||0)+1)+" done ✓");
  }

  function addCustom(){
    if(!cName.trim()){setCErr("Name required");return;}
    if(!cCode.trim()){setCErr("Code required");return;}
    const fn=makeCustomFn(cCode);if(!fn){setCErr("Syntax error — use: (s,M) => [result]");return;}
    try{fn([50,60,70,80]);}catch(e){setCErr("Runtime error: "+e.message);return;}
    const name=cName.trim().replace(/\s+/g,"_");
    if((S.customs||[]).some(a=>a.name===name)){setCErr("Name exists");return;}
    upd(prev=>({...prev,customs:[...(prev.customs||[]),{name,code:cCode,desc:"User-defined",enabled:true,generated:false,createdAt:Date.now()}]}));
    setCCode("");setCName("");setCErr("");st("\""+name+"\" saved ✓");
  }

  function toggleCustom(name){upd(prev=>({...prev,customs:(prev.customs||[]).map(a=>a.name===name?{...a,enabled:!a.enabled}:a)}));}
  function deleteCustom(name){if(!confirm("Delete \""+name+"\"?"))return;upd(prev=>({...prev,customs:(prev.customs||[]).filter(a=>a.name!==name)}));}
  function addDataset(){
    const name=dsName.trim()||"Dataset "+(Object.keys(S.datasets||{}).length+1);
    const id="ds_"+Date.now();
    upd(prev=>({...prev,datasets:{...(prev.datasets||{}),[id]:{name,rows:[]}},active:id}));
    setDsName("");
  }
  function resetAll(){if(!confirm("Clear ALL data?"))return;const s=fresh();setS(s);saveS(s);setCheckRes(null);setActs({A:"",B:"",C:"",D:""});setWfRes(null);setCorrM(null);st("Reset","warn");}

  const accLog=S.accLog||[];
  // useMemo: avoid recomputing on every keystroke/input change
  const {totalExact,overallPct}=useMemo(()=>{
    const te=accLog.reduce((s,e)=>s+(e.exactCount||0),0);
    return{totalExact:te,overallPct:accLog.length?Math.round(te/(accLog.length*4)*100):0};
  },[accLog]);
  function handleActChange(col,val){
    const v=val.toUpperCase();
    const cleaned=v==="X"||v==="XX"?"XX":v.replace(/\D/g,"").slice(0,2);
    setActs(p=>({...p,[col]:cleaned}));
  }
  const stClr=msg.c==="ok"?"#34d399":msg.c==="err"?"#f87171":msg.c==="warn"?"#fbbf24":msg.c==="busy"?"#a78bfa":"#4a4e6a";
  const dsKeys=Object.keys(S.datasets||{});
  const customs=S.customs||[];

  const rowDifficulty=useMemo(()=>computeRowDifficulty(accLog),[accLog]);
  const predRowDiff=S.predRow&&rowDifficulty[S.predRow]!=null?rowDifficulty[S.predRow]:null;

  const streaks=useMemo(()=>{
    const s={};
    COLS.forEach(col=>{
      let n=0;
      for(let i=accLog.length-1;i>=0;i--){
        if(accLog[i].results&&accLog[i].results[col]&&accLog[i].results[col].exact)n++;
        else break;
      }
      s[col]=n;
    });
    return s;
  },[accLog]);

  const missingRows=useMemo(()=>{
    const nums=rows.map(r=>r.row).sort((a,b)=>a-b);
    const missing=[];
    // Only flag gaps of ≤5 to avoid false positives when switching months
    for(let i=1;i<nums.length;i++){
      const gap=nums[i]-nums[i-1];
      if(gap>1&&gap<=5)for(let j=nums[i-1]+1;j<nums[i];j++)missing.push(j);
    }
    return missing.slice(0,20); // cap display at 20
  },[rows]);

  if(!loaded)return React.createElement("div",{style:{background:"#060709",minHeight:"100vh",display:"flex",alignItems:"center",justifyContent:"center",color:"#4a4e6a",fontFamily:"monospace"}},"Loading…");

  return (
    <div style={{background:"#060709",color:"#c8d0e8",minHeight:"100vh",fontFamily:"'Courier New',monospace",fontSize:13,backgroundImage:"radial-gradient(ellipse at 15% 0%,rgba(124,109,250,.07) 0%,transparent 55%),radial-gradient(ellipse at 85% 100%,rgba(52,211,153,.05) 0%,transparent 55%)"}}>
      <div style={{maxWidth:1100,margin:"0 auto",padding:"14px 12px 200px"}}>

        <div style={{textAlign:"center",padding:"18px 0 10px"}}>
          <div style={{fontSize:9,letterSpacing:5,color:"#252840",marginBottom:5,textTransform:"uppercase"}}>Self-Learning · Adaptive · Prediction · Engine</div>
          <div style={{fontSize:"clamp(28px,6vw,48px)",fontWeight:900,letterSpacing:-2,lineHeight:1,background:"linear-gradient(135deg,#a78bfa,#c4b5fd 35%,#34d399 70%,#6ee7b7)",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent",backgroundClip:"text"}}>APE v16</div>
          <div style={{fontSize:9,color:"#252840",marginTop:4}}>{ALGO_COUNT} built-in · cross-period pattern bank · global series calibration · multi-layer harmony · cross-dataset anniversary · auto-gen · live terminal</div>
          {accLog.length>0&&<div style={{marginTop:8,display:"inline-flex",gap:8,alignItems:"center",background:"rgba(52,211,153,.07)",border:"1px solid rgba(52,211,153,.18)",borderRadius:99,padding:"3px 14px",fontSize:10,color:"#34d399"}}>🧠 {accLog.length} sessions · {overallPct}% exact · {customs.length} algos</div>}
        </div>

        <div style={{display:"flex",gap:6,padding:"6px 0 10px",overflowX:"auto",flexWrap:"wrap",alignItems:"center"}}>
          {dsKeys.map(id=><button key={id} onClick={()=>upd(prev=>({...prev,active:id}))} style={{background:S.active===id?"rgba(167,139,250,.15)":"transparent",border:"1px solid "+(S.active===id?"rgba(167,139,250,.4)":"#1a1e35"),color:S.active===id?"#a78bfa":"#4a4e6a",padding:"3px 10px",borderRadius:99,cursor:"pointer",fontSize:10,fontFamily:"inherit",whiteSpace:"nowrap"}}>{S.datasets[id]?S.datasets[id].name:"?"} ({S.datasets[id]&&S.datasets[id].rows?S.datasets[id].rows.length:0})</button>)}
          <input value={dsName} onChange={e=>setDsName(e.target.value)} placeholder="New dataset…" style={{background:"#060709",border:"1px solid #1a1e35",color:"#c8d0e8",padding:"3px 8px",borderRadius:99,fontSize:10,fontFamily:"monospace",outline:"none",width:100}}/>
          <GB onClick={addDataset}>＋</GB>
        </div>

        <div style={{display:"flex",borderBottom:"1px solid #12152a",marginBottom:14,overflowX:"auto"}}>
          {[{id:"data",l:"📊 Data ("+rows.length+")"},{id:"predict",l:"🔮 Predict"},{id:"learn",l:"✅ Learn"+(accLog.length?" ("+accLog.length+")":"")},{id:"analysis",l:"📈 Analysis"},{id:"algos",l:"⚙ Algos ("+(ALGO_COUNT+customs.length)+")"}].map(function(t){
            return <button key={t.id} onClick={()=>setTab(t.id)} style={{background:tab===t.id?"rgba(167,139,250,.1)":"transparent",border:"none",borderBottom:tab===t.id?"2px solid #a78bfa":"2px solid transparent",color:tab===t.id?"#a78bfa":"#4a4e6a",padding:"8px 12px",cursor:"pointer",fontSize:11,fontFamily:"inherit",whiteSpace:"nowrap",marginBottom:-1}}>{t.l}</button>;
          })}
          <div style={{flex:1}}/>
          <div style={{display:"flex",gap:6,alignItems:"center",alignSelf:"center",marginRight:4}}>
            <div style={{fontSize:8,color:"#2d3158",letterSpacing:1}}>PRED DATE</div>
            <input type="date" value={predDate} onChange={e=>setPredDate(e.target.value)} style={{background:"#060709",border:"1px solid #1a1e35",color:predDate?"#a78bfa":"#2d3158",padding:"4px 6px",borderRadius:6,fontSize:10,fontFamily:"monospace",outline:"none",width:120,colorScheme:"dark"}}/>
          </div>
          <button onClick={runPredict} style={{background:"linear-gradient(135deg,#7c3aed,#a78bfa)",border:"none",color:"#fff",padding:"7px 16px",borderRadius:7,cursor:"pointer",fontSize:11,fontWeight:700,fontFamily:"inherit",alignSelf:"center",marginRight:2}}>🔮 Predict</button>
        </div>

        {tab==="data"&&<div>
          <Card>
            <SL>Add Row — {S.datasets[S.active]?S.datasets[S.active].name:"?"}</SL>
            <div style={{display:"flex",gap:8,flexWrap:"wrap",alignItems:"flex-end"}}>
              <FI label="Day #" val={rowIn} onChange={setRowIn} w={62} maxLen={4}/>
              <div>
                <div style={{fontSize:9,color:"#a78bfa",marginBottom:3,letterSpacing:2}}>DATE</div>
                <input type="date" value={dateIn} onChange={e=>setDateIn(e.target.value)} style={{background:"#060709",border:"1px solid #1a1e35",color:"#c8d0e8",padding:"7px 6px",borderRadius:6,fontSize:11,fontFamily:"monospace",outline:"none",width:130,colorScheme:"dark"}}/>
              </div>
              <FI label="A" val={vals.A} color={CLR.A} onChange={v=>setVals(p=>({...p,A:v}))} w={54}/>
              <FI label="B" val={vals.B} color={CLR.B} onChange={v=>setVals(p=>({...p,B:v}))} w={54}/>
              <FI label="C" val={vals.C} color={CLR.C} onChange={v=>setVals(p=>({...p,C:v}))} w={54}/>
              <FI label="D" val={vals.D} color={CLR.D} onChange={v=>setVals(p=>({...p,D:v}))} w={54}/>
              <PB onClick={addRow}>＋ Add</PB>
              <GB onClick={()=>{setRowIn(String(new Date().getDate()).padStart(2,"0"));setDateIn(todayStr());}}>📅 Today</GB>
            </div>
            {dateIn&&parseDate(dateIn)&&<DateBadge pd={parseDate(dateIn)}/>}
            {missingRows.length>0&&<div style={{marginTop:8,fontSize:10,color:"#fbbf24",background:"rgba(251,191,36,.06)",border:"1px solid rgba(251,191,36,.18)",borderRadius:5,padding:"4px 10px"}}>⚠ Missing rows: {missingRows.map(r=>pad2(r)).join(", ")}</div>}
          </Card>
          <div style={{marginBottom:12}}>
            {/* ── AUTO-TRAIN CARD ── */}
            <div style={{background:"linear-gradient(135deg,rgba(167,139,250,.08),rgba(52,211,153,.05))",border:"1px solid rgba(167,139,250,.25)",borderRadius:10,padding:"12px 14px",marginBottom:10}}>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",flexWrap:"wrap",gap:8}}>
                <div>
                  <div style={{fontSize:11,fontWeight:700,color:"#a78bfa",marginBottom:3}}>⚡ Auto-Train on Historical Data</div>
                  <div style={{fontSize:10,color:"#4a4e6a",lineHeight:1.6,maxWidth:480}}>
                    Upload a CSV of <b style={{color:"#c8d0e8"}}>past data with known outcomes</b>. APE simulates predictions row-by-row in time order, compares each prediction to the actual, updates weights, prunes weak algos, and builds the pattern bank — all automatically.
                  </div>
                  <div style={{marginTop:8,fontSize:9,color:"#2d3158",fontFamily:"monospace",background:"rgba(0,0,0,.3)",borderRadius:5,padding:"5px 8px",display:"inline-block",lineHeight:1.8}}>
                    <span style={{color:"#fbbf24"}}>CSV Format:</span><br/>
                    Row,A,B,C,D,Date<br/>
                    <span style={{color:"#4a4e6a"}}>1,42,87,13,65,2025-01-01</span><br/>
                    <span style={{color:"#4a4e6a"}}>2,91,10,55,30,2025-01-02</span><br/>
                    <span style={{color:"#34d399"}}>• Date column optional but strongly recommended</span><br/>
                    <span style={{color:"#34d399"}}>• Use XX for unknown columns</span><br/>
                    <span style={{color:"#34d399"}}>• Row = day number (any integer, no cycling)</span>
                  </div>
                </div>
                <div style={{display:"flex",flexDirection:"column",gap:6}}>
                  <label style={{background:"linear-gradient(135deg,#7c3aed,#a78bfa)",border:"none",color:"#fff",padding:"10px 18px",borderRadius:7,cursor:"pointer",fontSize:12,fontWeight:700,fontFamily:"inherit",textAlign:"center",lineHeight:1.4}}>
                    🚀 Upload & Auto-Train
                    <span style={{fontSize:9,opacity:.8,display:"block"}}>CSV with known outcomes</span>
                    <input type="file" accept=".csv,.txt" onChange={doAutoTrain} style={{display:"none"}}/>
                  </label>
                  {autoTrainStatus?.running&&<button onClick={()=>{autoTrainRef.current=true;}} style={{background:"rgba(248,113,113,.15)",border:"1px solid rgba(248,113,113,.3)",color:"#f87171",padding:"6px 14px",borderRadius:6,cursor:"pointer",fontSize:11,fontFamily:"inherit"}}>⛔ Stop</button>}
                </div>
              </div>
              {/* Progress */}
              {autoTrainStatus&&<div style={{marginTop:10}}>
                {autoTrainStatus.running&&<div style={{marginBottom:6}}>
                  <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:"#4a4e6a",marginBottom:3}}>
                    <span>Training…</span>
                    <span>{autoTrainStatus.progress}/{autoTrainStatus.total} rows</span>
                  </div>
                  <div style={{height:4,background:"#1a1e35",borderRadius:99}}>
                    <div style={{height:"100%",width:Math.round((autoTrainStatus.progress/autoTrainStatus.total||0)*100)+"%",background:"linear-gradient(90deg,#7c3aed,#34d399)",borderRadius:99,transition:"width .3s"}}/>
                  </div>
                </div>}
                {autoTrainStatus.result&&<div style={{display:"flex",gap:10,flexWrap:"wrap",marginBottom:6}}>
                  <div style={{background:"rgba(52,211,153,.08)",border:"1px solid rgba(52,211,153,.2)",borderRadius:6,padding:"4px 10px",fontSize:10}}>
                    <span style={{color:"#34d399",fontWeight:700}}>{autoTrainStatus.result.exactPct}%</span> <span style={{color:"#4a4e6a"}}>exact</span>
                  </div>
                  <div style={{background:"rgba(167,139,250,.08)",border:"1px solid rgba(167,139,250,.2)",borderRadius:6,padding:"4px 10px",fontSize:10}}>
                    <span style={{color:"#a78bfa",fontWeight:700}}>{autoTrainStatus.result.nearPct}%</span> <span style={{color:"#4a4e6a"}}>exact+near</span>
                  </div>
                  <div style={{background:"rgba(251,191,36,.06)",border:"1px solid rgba(251,191,36,.15)",borderRadius:6,padding:"4px 10px",fontSize:10}}>
                    <span style={{color:"#fbbf24",fontWeight:700}}>{autoTrainStatus.result.kept}</span> <span style={{color:"#4a4e6a"}}>algos kept</span>
                  </div>
                  <div style={{background:"rgba(248,113,113,.06)",border:"1px solid rgba(248,113,113,.15)",borderRadius:6,padding:"4px 10px",fontSize:10}}>
                    <span style={{color:"#f87171",fontWeight:700}}>{autoTrainStatus.result.pruned}</span> <span style={{color:"#4a4e6a"}}>pruned</span>
                  </div>
                  {autoTrainStatus.result.perf&&<div style={{background:"rgba(96,165,250,.08)",border:"1px solid rgba(96,165,250,.2)",borderRadius:6,padding:"4px 10px",fontSize:10}}>
                    <span style={{color:"#60a5fa",fontWeight:700}}>{autoTrainStatus.result.perf.rowsPerSec}</span> <span style={{color:"#4a4e6a"}}>rows/s · {autoTrainStatus.result.perf.avgPredMs}ms pred</span>
                  </div>}
                </div>}
                <div style={{background:"#060709",border:"1px solid #12152a",borderRadius:6,padding:"6px 10px",maxHeight:120,overflowY:"auto",fontFamily:"monospace",fontSize:9,lineHeight:1.7}}>
                  {(autoTrainStatus.log||[]).map((l,i)=><div key={i} style={{color:l.startsWith("✅")||l.startsWith("🎯")?"#34d399":l.startsWith("❌")||l.startsWith("⛔")?"#f87171":l.startsWith("📊")?"#fbbf24":l.startsWith("🧠")?"#a78bfa":"#4a4e6a"}}>{l}</div>)}
                </div>
              </div>}
            </div>

            <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
              <GB onClick={()=>setShowBulk(!showBulk)}>{showBulk?"▲":"▼"} Bulk Paste</GB>
              <GB onClick={()=>doExportCSV(rows,S.preds,S.predRow)}>📥 Export CSV</GB>
              <GB onClick={()=>doExportJSON(S)}>📦 Export JSON</GB>
              <label style={{background:"transparent",border:"1px solid #1a1e35",color:"#8892b0",padding:"7px 12px",borderRadius:6,cursor:"pointer",fontSize:11,fontFamily:"inherit"}}>📂 Import JSON<input type="file" accept=".json" onChange={doImport} style={{display:"none"}}/></label>
              <button onClick={doRebuildPatternBank} style={{background:"rgba(167,139,250,.08)",border:"1px solid rgba(167,139,250,.25)",color:"#a78bfa",padding:"7px 12px",borderRadius:6,cursor:"pointer",fontSize:11,fontFamily:"inherit"}}>
                🧠 Rebuild Patterns ({COLS.reduce((s,c)=>s+Object.keys((S.patternBank||{})[c]?.mdProfiles||{}).length,0)} stored)
              </button>
            </div>
            {showBulk&&<div style={{marginTop:8,background:"#0c0e1a",border:"1px solid #1a1e35",borderRadius:8,padding:12}}>
              <textarea value={bulk} onChange={e=>setBulk(e.target.value)} placeholder={"01,02,10,92,XX\n02,91,10,30,68"} style={{width:"100%",height:90,background:"#060709",border:"1px solid #1a1e35",color:"#c8d0e8",padding:8,borderRadius:6,fontSize:11,resize:"vertical",fontFamily:"monospace",outline:"none",boxSizing:"border-box"}}/>
              <div style={{display:"flex",gap:6,marginTop:8}}><PB onClick={doBulk}>Import</PB><GB onClick={()=>setShowBulk(false)}>Cancel</GB></div>
            </div>}
          </div>
          {rows.length>0?<Card>
            <div style={{overflowX:"auto",maxHeight:320,overflowY:"auto"}}>
              <table style={{width:"100%",borderCollapse:"collapse",fontSize:12}}>
                <thead><tr style={{background:"#0c0e1a",position:"sticky",top:0}}>{["#","Row","Date","A","B","C","D",""].map((h,i)=><th key={i} style={{padding:"6px 10px",color:i===2?"#a78bfa":"#252840",fontSize:9,letterSpacing:2,textTransform:"uppercase",borderBottom:"1px solid #1a1e35",textAlign:"center"}}>{h}</th>)}</tr></thead>
                <tbody>{rows.map((r,i)=>{const dp=r.date?parseDate(r.date):null;const DAYS=["Su","Mo","Tu","We","Th","Fr","Sa"];return<tr key={r.row} style={{background:i%2?"rgba(255,255,255,.01)":"transparent",borderBottom:"1px solid rgba(255,255,255,.02)"}}>
                  <td style={{padding:"5px 10px",color:"#252840",textAlign:"center"}}>{i+1}</td>
                  <td style={{padding:"5px 10px",color:"#fbbf24",fontWeight:700,textAlign:"center"}}>{pad2(r.row)}</td>
                  <td style={{padding:"5px 8px",textAlign:"center",fontSize:9}}>{dp?<span style={{color:"#a78bfa"}}>{r.date.slice(5)} <span style={{color:"#2d3158"}}>{DAYS[dp.dow]}</span></span>:<span style={{color:"#1a1e35"}}>—</span>}</td>
                  {COLS.map(col=><td key={col} style={{padding:"5px 10px",textAlign:"center",fontWeight:700,color:r[col]===null?"#1a1e35":CLR[col]}}>{r[col]===null?"—":pad2(r[col])}</td>)}
                  <td style={{padding:"5px 8px",textAlign:"center"}}><button onClick={()=>{setRows(prev=>prev.filter(x=>x.row!==r.row));st("Row "+r.row+" deleted","warn");}} style={{background:"transparent",border:"none",color:"#252840",cursor:"pointer",fontSize:11}}>✕</button></td>
                </tr>;})}</tbody>
              </table>
            </div>
            <div style={{padding:"7px 12px",fontSize:9,color:"#252840",borderTop:"1px solid #1a1e35",display:"flex",justifyContent:"space-between",alignItems:"center"}}>
              <span>{rows.length} rows · auto-saved</span>
              <button onClick={resetAll} style={{background:"transparent",border:"none",color:"#f87171",cursor:"pointer",fontSize:9,fontFamily:"inherit"}}>🗑 Clear All</button>
            </div>
          </Card>:<div style={{textAlign:"center",padding:"40px 0",color:"#1e2240",fontSize:12}}>No data yet.</div>}
        </div>}

        {tab==="predict"&&<div>
          {!S.preds?<div style={{textAlign:"center",padding:"40px 0"}}>
            <div style={{color:"#2d3158",marginBottom:16}}>No predictions yet.</div>
            <PB onClick={runPredict}>🔮 Run Prediction ({rows.length} rows)</PB>
          </div>:<div>
            <div style={{background:"rgba(167,139,250,.06)",border:"1px solid rgba(167,139,250,.2)",borderRadius:10,padding:"12px 16px",marginBottom:14,display:"flex",alignItems:"center",gap:16,flexWrap:"wrap"}}>
              <div>
                <div style={{fontSize:9,color:"#4a4e6a",letterSpacing:2,marginBottom:2}}>PREDICTING ROW</div>
                <div style={{fontSize:44,fontWeight:900,color:"#a78bfa",lineHeight:1,fontFamily:"monospace"}}>{pad2(S.predRow||0)}</div>
                {predRowDiff!=null&&<div style={{fontSize:9,marginTop:3,fontWeight:700,color:predRowDiff>40?"#34d399":predRowDiff>20?"#fbbf24":"#f87171"}}>{predRowDiff>40?"🟢 Easy row":predRowDiff>20?"🟡 Medium row":"🔴 Hard row"} ({predRowDiff}% accuracy)</div>}
                {S.predDate&&parseDate(S.predDate)&&<DateBadge pd={parseDate(S.predDate)} showDate={S.predDate} fullRow/>}
              </div>
              <div style={{flex:1,minWidth:200}}>
                <div style={{fontSize:9,color:"#4a4e6a",letterSpacing:2,marginBottom:6}}>TOP PICKS</div>
                <div style={{display:"flex",gap:10,flexWrap:"wrap"}}>
                  {COLS.map(col=>{
                    const top=S.preds[col]&&S.preds[col].top5[0]?S.preds[col].top5[0]:null;
                    const confClr=S.preds[col]?S.preds[col].confClr:"#4a4e6a";
                    const conf=S.preds[col]?S.preds[col].conf:"?";
                    const regime=S.preds[col]?S.preds[col].regime:"normal";
                    if(!top)return null;
                    return <div key={col} style={{textAlign:"center"}}>
                      <div style={{fontSize:9,color:CLR[col],letterSpacing:2,marginBottom:2}}>{col}</div>
                      <div style={{fontSize:24,fontWeight:900,color:CLR[col],background:CLR[col]+"18",border:"1px solid "+CLR[col]+"44",borderRadius:8,padding:"4px 10px",minWidth:52,textAlign:"center"}}>{pad2(top.value)}</div>
                      <div style={{fontSize:8,color:"#4a4e6a",marginTop:2}}>{top.pct}%</div>
                      <div style={{fontSize:8,color:confClr,fontWeight:700}}>{conf}</div>
                      {regime!=="normal"&&<div style={{fontSize:7,color:"#fbbf24"}}>{regime}</div>}
                      {streaks[col]>=2&&<div style={{fontSize:8,color:"#34d399"}}>🔥{streaks[col]}</div>}
                      {S.preds[col]&&S.preds[col].familyAgreement>=3&&<div style={{fontSize:7,color:"#a78bfa",marginTop:1}}>{"★".repeat(Math.min(S.preds[col].familyAgreement,5))} cross-family</div>}
                    </div>;
                  })}
                </div>
              </div>
              <div style={{display:"flex",flexDirection:"column",gap:6,alignItems:"flex-end"}}>
                <div style={{textAlign:"center"}}>
                  <div style={{fontSize:9,color:"#4a4e6a"}}>ALGOS</div>
                  <div style={{fontSize:20,fontWeight:700}}>{S.preds.A?S.preds.A.algoCount:0}</div>
                </div>
                <button onClick={()=>{
                  const lines=["APE v13 — Row "+pad2(S.predRow||0)+" — "+new Date().toLocaleString(),"─".repeat(36)];
                  COLS.forEach(col=>{const t=S.preds[col]?S.preds[col].top5:[];lines.push("Col "+col+": "+t.map((p,i)=>(i===0?"▶":"")+pad2(p.value)+"("+p.pct+"%)").join("  "));});
                  lines.push("─".repeat(36),"Generated by APE v13");
                  navigator.clipboard&&navigator.clipboard.writeText(lines.join("\n")).catch(()=>{});
                  setCopyMsg("Copied!");setTimeout(()=>setCopyMsg(""),2000);
                }} style={{background:"rgba(167,139,250,.1)",border:"1px solid rgba(167,139,250,.3)",color:"#a78bfa",padding:"5px 10px",borderRadius:6,cursor:"pointer",fontSize:9,fontFamily:"inherit"}}>{copyMsg||"📋 Copy"}</button>
              </div>
            </div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(216px,1fr))",gap:12,marginBottom:14}}>
              {COLS.map(col=>{
                const pred=S.preds[col];if(!pred)return null;
                const clr=CLR[col],maxV=pred.top5[0]?pred.top5[0].votes:1;
                return <div key={col} style={{background:"#0c0e1a",border:"1px solid #1a1e35",borderTop:"3px solid "+clr,borderRadius:10,padding:14}}>
                  <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:6}}>
                    <span style={{fontSize:20,fontWeight:900,color:clr}}>Col {col}</span>
                    <div style={{textAlign:"right"}}>
                      <div style={{fontSize:8,fontWeight:700,color:pred.confClr}}>{pred.conf}</div>
                      <div style={{fontSize:8,color:"#2d3158"}}>spread:{pred.variance} · {pred.regime}</div>
                    </div>
                  </div>
                  <div style={{fontSize:9,color:"#2d3158",marginBottom:6}}>band: <span style={{color:"#a78bfa"}}>{pad2(pred.bandLo||0)}–{pad2(pred.bandHi||0)}</span> · consensus:{pred.consensus}%</div>
                  <div style={{marginBottom:10,height:3,background:"#1a1e35",borderRadius:99}}><div style={{height:"100%",width:pred.consensus+"%",background:pred.consensus>50?"#34d399":pred.consensus>25?"#fbbf24":"#f87171",borderRadius:99}}/></div>
                  {/* ── TEMPORAL SIGNAL STRIP ── */}
                  {pred.tempTotal>0&&<div style={{marginBottom:8,padding:"5px 8px",background:"rgba(52,211,153,.04)",border:"1px solid rgba(52,211,153,.12)",borderRadius:6}}>
                    <div style={{fontSize:8,color:"#34d399",letterSpacing:2,marginBottom:4}}>⏱ TEMPORAL · {pred.tempAgree}/{pred.tempTotal} signals agree</div>
                    <div style={{display:"flex",flexWrap:"wrap",gap:3}}>
                      {pred.tempSignals.slice(0,6).map(s=><span key={s.name} style={{
                        fontSize:8,padding:"2px 5px",borderRadius:4,fontFamily:"monospace",
                        background:s.match?"rgba(52,211,153,.15)":"rgba(255,255,255,.03)",
                        border:"1px solid "+(s.match?"rgba(52,211,153,.4)":"rgba(255,255,255,.06)"),
                        color:s.match?"#34d399":"#2d3158"
                      }}>{pad2(s.pred)}</span>)}
                    </div>
                  </div>}
                  {/* ── DATE SIGNAL STRIP ── */}
                  {pred.dateTotal>0&&<div style={{marginBottom:8,padding:"5px 8px",background:"rgba(167,139,250,.04)",border:"1px solid rgba(167,139,250,.12)",borderRadius:6}}>
                    <div style={{fontSize:8,color:"#a78bfa",letterSpacing:2,marginBottom:4}}>📅 DATE · {pred.dateAgree}/{pred.dateTotal} signals agree</div>
                    <div style={{display:"flex",flexWrap:"wrap",gap:3}}>
                      {pred.dateSigList.slice(0,8).map(s=><span key={s.name} title={s.name} style={{
                        fontSize:8,padding:"2px 5px",borderRadius:4,fontFamily:"monospace",cursor:"default",
                        background:s.match?"rgba(167,139,250,.18)":"rgba(255,255,255,.03)",
                        border:"1px solid "+(s.match?"rgba(167,139,250,.45)":"rgba(255,255,255,.06)"),
                        color:s.match?"#c4b5fd":"#2d3158"
                      }}>{pad2(s.pred)}<span style={{fontSize:6,opacity:.6,marginLeft:2}}>{s.name.split("_")[0]}</span></span>)}
                    </div>
                  </div>}
                  {pred.top5.map((p,i)=><div key={i} style={{marginBottom:i<4?6:0}}>
                    <div style={{display:"flex",alignItems:"center",gap:5}}>
                      <span style={{fontWeight:700,fontSize:i===0?20:12,color:i===0?clr:i===1?clr+"88":"#2d3158",minWidth:i===0?36:26,fontFamily:"monospace"}}>{i===0?"▶":" "+(i+1)} {pad2(p.value)}</span>
                      <div style={{flex:1,height:i===0?5:2,background:"#1a1e35",borderRadius:99,overflow:"hidden"}}><div style={{height:"100%",width:Math.round(p.votes/maxV*100)+"%",background:i===0?clr:i===1?clr+"55":"#1a1e35",borderRadius:99}}/></div>
                      <span style={{fontSize:9,color:"#2d3158",minWidth:24,textAlign:"right"}}>{p.pct}%</span>
                    </div>
                    {i===0&&<div style={{fontSize:9,color:"#2d3158",marginLeft:36,marginTop:1,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",maxWidth:155}}>{p.algos.slice(0,5).join(" · ")}</div>}
                  </div>)}
                  <button onClick={()=>setExpCol(expCol===col?null:col)} style={{marginTop:10,width:"100%",background:"transparent",border:"1px solid #1a1e35",color:"#2d3158",borderRadius:6,padding:"4px 0",fontSize:9,cursor:"pointer",fontFamily:"inherit"}}>{expCol===col?"▲ Hide":"▼ All algo votes"}</button>
                  {expCol===col&&<div style={{marginTop:8,maxHeight:260,overflowY:"auto"}}>
                    <div style={{display:"grid",gridTemplateColumns:"1fr auto auto auto auto",gap:2,fontSize:9,color:"#2d3158",borderBottom:"1px solid #1a1e35",paddingBottom:3,marginBottom:3}}>
                      <span>Algo</span><span>Pred</span><span>BT%</span><span>Gw</span><span>Rw</span>
                    </div>
                    {Object.entries(pred.details).sort((a,b)=>b[1].w-a[1].w).map(([name,info])=>{
                      const inTop=pred.top5.some(t=>t.value===info.pred&&t.algos.includes(name));
                      const tc=info.type==="date"?"#a78bfa":info.type==="temporal"?"#34d399":info.type==="rowhistory"?"#f59e0b":info.type==="colgap"?"#38bdf8":info.type==="custom"?"#f87171":info.type==="cross"?"#fbbf24":"#4a4e6a";
                      return <div key={name} style={{display:"grid",gridTemplateColumns:"1fr auto auto auto auto",gap:3,padding:"2px 0",borderBottom:"1px solid rgba(255,255,255,.015)",background:inTop?clr+"08":"transparent",fontSize:9}}>
                        <span style={{color:tc,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{name}</span>
                        <span style={{color:inTop?clr:"#4a4e6a",fontWeight:inTop?700:400,textAlign:"center"}}>{pad2(info.pred)}</span>
                        <span style={{color:info.bt===null?"#fbbf24":info.bt>20?"#34d399":info.bt>7?"#fbbf24":"#2d3158",textAlign:"right"}}>{info.bt===null?"⊕":info.bt+"%"}</span>
                        <span style={{color:info.lw>1.5?"#34d399":info.lw<0.5?"#f87171":"#4a4e6a",textAlign:"right"}}>{info.lw}×</span>
                        <span style={{color:info.rw>1.2?"#34d399":info.rw<0.8?"#f87171":"#4a4e6a",textAlign:"right"}}>{info.rw}×</span>
                      </div>;
                    })}
                  </div>}
                </div>;
              })}
            </div>
            <Card style={{marginBottom:14}}>
              <SL>Top-5 Table — Row {pad2(S.predRow||0)}</SL>
              <div style={{overflowX:"auto"}}>
                <table style={{borderCollapse:"collapse",width:"100%",fontSize:13}}>
                  <thead><tr>
                    <th style={{padding:"5px 12px",color:"#2d3158",fontSize:9,borderBottom:"1px solid #1a1e35",textAlign:"left"}}>Rank</th>
                    {COLS.map(c=><th key={c} style={{padding:"5px 12px",color:CLR[c],fontSize:14,borderBottom:"1px solid #1a1e35",textAlign:"center",fontWeight:900}}>{c}</th>)}
                  </tr></thead>
                  <tbody>{[0,1,2,3,4].map(rank=><tr key={rank} style={{background:rank===0?"rgba(167,139,250,.05)":"transparent"}}>
                    <td style={{padding:"5px 12px",color:rank===0?"#a78bfa":"#2d3158",fontSize:rank===0?11:9,fontWeight:rank===0?700:400}}>{rank===0?"► #1 PICK":"  #"+(rank+1)}</td>
                    {COLS.map(col=><PredCell key={col} col={col} rank={rank} preds={S.preds}/>)}
                  </tr>)}</tbody>
                </table>
              </div>
            </Card>
            <GB onClick={()=>setTab("learn")}>✅ Enter Actual and Learn →</GB>
          </div>}
        </div>}

        {tab==="learn"&&<div>
          <Card style={{marginBottom:14}}>
            <SL>{S.preds?"Enter Actual for Row "+pad2(S.predRow||0):"Run prediction first"}</SL>
            {S.preds&&<div>
              <div style={{marginBottom:8,fontSize:10,color:"#4a4e6a",background:"rgba(167,139,250,.04)",border:"1px solid rgba(167,139,250,.12)",borderRadius:6,padding:"6px 10px",lineHeight:1.7}}>
                💡 Enter values you know. Type <b style={{color:"#a78bfa"}}>XX</b> or leave blank for unknown columns. APE learns only from known values and fills missing ones from historical patterns.
              </div>
              <div style={{display:"flex",gap:10,flexWrap:"wrap",alignItems:"flex-end",marginBottom:10}}>
                {COLS.map(col=><ActInput key={col} col={col} val={acts[col]} pred={S.preds[col]} onChg={handleActChange}/>)}
              </div>
              <div style={{display:"flex",gap:8,flexWrap:"wrap",marginBottom:14}}>
                <PB onClick={checkAndLearn}>✅ Check and Learn</PB>
                <GB onClick={predictMissingCols}>🔮 Predict Missing Cols</GB>
                <GB onClick={()=>{setActs({A:"",B:"",C:"",D:""});setMissingPreds({});}}>✕ Clear All</GB>
              </div>
              {Object.keys(missingPreds).length>0&&<div style={{background:"rgba(167,139,250,.06)",border:"1px solid rgba(167,139,250,.2)",borderRadius:8,padding:12,marginBottom:12}}>
                <div style={{fontSize:9,color:"#a78bfa",letterSpacing:3,textTransform:"uppercase",marginBottom:10}}>Predictions for Missing Columns</div>
                <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(160px,1fr))",gap:10}}>
                  {Object.entries(missingPreds).map(([col,pred])=>{
                    const maxV=pred.top5[0]?pred.top5[0].votes:1;
                    return <MissingColPred key={col} col={col} pred={pred} maxV={maxV}/>;
                  })}
                </div>
                <div style={{fontSize:9,color:"#2d3158",marginTop:8}}>These predictions used known column values as cross-col hints. Top pick is auto-filled in the input above.</div>
              </div>}
              {checkRes&&<div style={{background:"rgba(52,211,153,.04)",border:"1px solid rgba(52,211,153,.15)",borderRadius:8,padding:12}}>
                <SL style={{color:"#34d399"}}>Result</SL>
                <div style={{display:"flex",gap:10,flexWrap:"wrap",marginBottom:10}}>
                  {COLS.map(col=><CheckCard key={col} col={col} res={checkRes[col]}/>)}
                </div>
                <div style={{fontSize:10,color:"#4a4e6a",lineHeight:1.8}}>
                  {checkRes&&Object.values(checkRes).some(r=>r.skipped)?
                    <span>Partial learn: weights updated only for known columns. Unknown columns kept at current weights.</span>:
                    <span>Full learn: Bayesian momentum + per-row + per-range weights updated. Exact×1.4, Near×1.1, Miss×0.8.</span>
                  }
                </div>
                <div style={{marginTop:6,fontSize:10,color:"#34d399",background:"rgba(52,211,153,.06)",border:"1px solid rgba(52,211,153,.2)",borderRadius:5,padding:"4px 10px"}}>
                  ✅ Row {pad2(S.predRow||0)} added to dataset.
                  {checkRes&&Object.values(checkRes).some(r=>r.skipped)?" Unknown columns stored as XX (blank) — fill them in Data tab when you get the values.":""}
                </div>
              </div>}
            </div>}
          </Card>
          {accLog.length>0&&<Card style={{marginBottom:14}}>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}>
              <SL style={{margin:0}}>History ({accLog.length})</SL>
              <span style={{fontSize:11,color:overallPct>30?"#34d399":overallPct>10?"#fbbf24":"#f87171",fontWeight:700}}>🎯 {overallPct}% exact</span>
            </div>
            <div style={{overflowX:"auto",maxHeight:240,overflowY:"auto"}}>
              <table style={{borderCollapse:"collapse",width:"100%",fontSize:11}}>
                <thead><tr>{["Time","Row","PA","PB","PC","PD","AA","AB","AC","AD","Hit"].map((h,i)=><th key={i} style={{padding:"4px 7px",color:"#252840",fontSize:9,borderBottom:"1px solid #1a1e35",textAlign:"center",whiteSpace:"nowrap"}}>{h}</th>)}</tr></thead>
                <tbody>{[...accLog].reverse().map((entry,i)=><tr key={i} style={{borderBottom:"1px solid rgba(255,255,255,.015)"}}>
                  <td style={{padding:"4px 7px",color:"#252840",fontSize:9,textAlign:"center"}}>{new Date(entry.at).toLocaleTimeString()}</td>
                  <td style={{padding:"4px 7px",color:"#fbbf24",fontWeight:700,textAlign:"center"}}>{pad2(entry.targetRow||0)}</td>
                  {COLS.map(c=><td key={c} style={{padding:"4px 7px",color:CLR[c],textAlign:"center",fontWeight:700}}>{pad2(entry.preds[c]||0)}</td>)}
                  {COLS.map(c=><td key={c} style={{padding:"4px 7px",color:entry.actuals[c]!=null?"#c8d0e8":"#2d3158",textAlign:"center"}}>{entry.actuals[c]!=null?pad2(entry.actuals[c]):"XX"}</td>)}
                  <td style={{padding:"4px 7px",textAlign:"center"}}><span style={{color:entry.exactCount>=(entry.knownCount||4)*0.75?"#34d399":entry.exactCount>=1?"#fbbf24":"#f87171",fontWeight:700}}>{entry.exactCount}/{entry.knownCount||4}</span></td>
                </tr>)}</tbody>
              </table>
            </div>
          </Card>}
          {Object.values(streaks).some(s=>s>0)&&<Card style={{marginBottom:14}}>
            <SL>🔥 Streak Tracker</SL>
            <div style={{display:"flex",gap:12,flexWrap:"wrap"}}>
              {COLS.map(col=><div key={col} style={{textAlign:"center",background:streaks[col]>=3?"rgba(52,211,153,.08)":"rgba(255,255,255,.02)",border:"1px solid "+(streaks[col]>=3?"rgba(52,211,153,.2)":"#1a1e35"),borderRadius:8,padding:"8px 14px"}}>
                <div style={{fontSize:9,color:CLR[col],letterSpacing:2,marginBottom:3}}>Col {col}</div>
                <div style={{fontSize:22,fontWeight:900,color:streaks[col]>=3?"#34d399":streaks[col]>=2?"#fbbf24":"#4a4e6a"}}>{streaks[col]}</div>
                <div style={{fontSize:9,color:"#2d3158"}}>exact in a row</div>
              </div>)}
            </div>
          </Card>}

          {S.calibration&&Object.values(S.calibration).some(c=>Object.values(c).some(x=>x.total>=3))&&<Card style={{marginBottom:14}}>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}>
              <SL style={{margin:0}}>🎯 Confidence Calibration</SL>
              <span style={{fontSize:9,color:"#4a4e6a"}}>How accurate are HIGH/MED/LOW labels?</span>
            </div>
            <div style={{display:"flex",gap:12,flexWrap:"wrap"}}>
              {COLS.map(col=><CalibCol key={col} col={col} cal={S.calibration[col]||{}}/>)}
            </div>
          </Card>}

          {accLog.length>0&&<Card>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}>
              <SL style={{margin:0}}>🧠 Neural Scores — Top Algos</SL>
              <span style={{fontSize:9,color:"#4a4e6a"}}>Running accuracy (positive = good)</span>
            </div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(210px,1fr))",gap:10}}>
              {COLS.map(col=><NeuralScoreCol key={col} col={col} scores={S.weights[col]?S.weights[col].neuralScores||{}:{}} customs={S.customs||[]}/>)}
            </div>
          </Card>}
        </div>}

        {tab==="analysis"&&<div>
          {rows.length>0&&<Card style={{marginBottom:14}}>
            <SL>Value Heatmap</SL>
            <div style={{overflowX:"auto"}}>
              <table style={{borderCollapse:"collapse",fontSize:11}}>
                <thead><tr><th style={{padding:"4px 8px",color:"#252840",fontSize:9,borderBottom:"1px solid #1a1e35",textAlign:"center"}}>Row</th>{COLS.map(c=><th key={c} style={{padding:"4px 12px",color:CLR[c],fontSize:11,borderBottom:"1px solid #1a1e35",textAlign:"center",fontWeight:900}}>{c}</th>)}</tr></thead>
                <tbody>{rows.map(r=><tr key={r.row}><td style={{padding:"3px 8px",color:"#fbbf24",fontWeight:700,textAlign:"center",fontSize:10}}>{pad2(r.row)}</td>
                  {COLS.map(col=><HeatCell key={col} col={col} v={r[col]}/>)}
                </tr>)}</tbody>
              </table>
            </div>
          </Card>}
          <Card style={{marginBottom:14}}>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}><SL style={{margin:0}}>Correlation Matrix</SL><GB onClick={()=>setCorrM(buildCorr(rows))}>Calculate</GB></div>
            {corrM?<CorrTable corrM={corrM}/>:<div style={{fontSize:11,color:"#2d3158"}}>Click Calculate to build correlation matrix.</div>}
          </Card>
          {rows.length>5&&<Card style={{marginBottom:14}}>
            <SL>Outlier Detection (more than 2σ)</SL>
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(200px,1fr))",gap:10}}>
              {COLS.map(col=><OutlierCol key={col} col={col} rows={rows}/>)}
            </div>
          </Card>}
          {rows.length>3&&<Card style={{marginBottom:14}}>
            <SL>Hot and Cold Numbers</SL>
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(200px,1fr))",gap:10}}>
              {COLS.map(col=><HotColdCol key={col} col={col} rows={rows}/>)}
            </div>
          </Card>}
          <Card>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}><SL style={{margin:0}}>Walk-Forward Test</SL><GB onClick={()=>{if(rows.length<8){st("Need at least 8 rows","warn");return;}const wf={};COLS.forEach(col=>{const series=getSeries(col,rows);const cr={};Object.entries(A).forEach(([name,fn])=>{const r=walkFwd(fn,series);if(r)cr[name]=r;});wf[col]=Object.entries(cr).sort((a,b)=>b[1].pct-a[1].pct).slice(0,8);});setWfRes(wf);st("Walk-forward done ✓");}}>Run Test</GB></div>
            {wfRes?<div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(220px,1fr))",gap:10}}>
              {COLS.map(col=><div key={col}><div style={{fontSize:9,color:CLR[col],letterSpacing:2,marginBottom:6}}>Best algos for {col}</div>
                {(wfRes[col]||[]).map(([name,r])=><div key={name} style={{display:"flex",gap:6,alignItems:"center",marginBottom:3,fontSize:9}}>
                  <span style={{flex:1,color:"#4a4e6a",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{name}</span>
                  <span style={{color:r.pct>60?"#34d399":r.pct>30?"#fbbf24":"#f87171",fontWeight:700}}>{r.pct}%</span>
                  <span style={{color:"#4a4e6a",fontSize:8}}>T1:{r.top1pct||0}%</span>
                </div>)}
              </div>)}
            </div>:<div style={{fontSize:11,color:"#2d3158"}}>Click Run Test. Needs at least 8 rows.</div>}
          </Card>
        </div>}

        {tab==="algos"&&<div>
          <div style={{background:"rgba(167,139,250,.06)",border:"1px solid rgba(167,139,250,.25)",borderRadius:10,padding:14,marginBottom:14}}>
            <SL style={{color:"#a78bfa"}}>💾 Save and Restore Training Progress</SL>
            <p style={{fontSize:11,color:"#4a4e6a",marginBottom:12,lineHeight:1.7}}>
              Export your learned weights, custom algorithms, and session history to a file. Import it back any time — your training progress is never lost even if storage clears.
            </p>
            <div style={{display:"flex",gap:8,flexWrap:"wrap",alignItems:"center"}}>
              <PB onClick={()=>doExportWeights(S)} style={{background:"#a78bfa"}}>📤 Export Weights</PB>
              <label style={{background:"transparent",border:"1px solid rgba(52,211,153,.3)",color:"#34d399",padding:"9px 18px",borderRadius:6,cursor:"pointer",fontSize:12,fontFamily:"inherit",fontWeight:700}}>
                📥 Import Weights<input type="file" accept=".json" onChange={doImportWeights} style={{display:"none"}}/>
              </label>
              <GB onClick={()=>doExportJSON(S)}>📦 Full Backup</GB>
            </div>
            {weightsMsg&&<div style={{marginTop:10,fontSize:11,color:weightsMsg.startsWith("✅")?"#34d399":"#f87171",background:weightsMsg.startsWith("✅")?"rgba(52,211,153,.06)":"rgba(248,113,113,.06)",border:"1px solid "+(weightsMsg.startsWith("✅")?"rgba(52,211,153,.2)":"rgba(248,113,113,.2)"),padding:"6px 10px",borderRadius:6}}>{weightsMsg}</div>}
            <div style={{marginTop:10,fontSize:9,color:"#2d3158",lineHeight:1.8}}>
              Weights file contains: global weights, per-row weights, per-range weights, custom/generated algos, and session history.<br/>
              After importing, run Predict immediately — no retraining needed. All algo vote powers restore instantly.
            </div>
          </div>

          {autoGenLog.length>0&&<div style={{background:"rgba(167,139,250,.06)",border:"1px solid rgba(167,139,250,.2)",borderRadius:8,padding:"8px 12px",marginBottom:10,fontSize:10}}>
            <div style={{color:"#a78bfa",fontWeight:700,marginBottom:4}}>🤖 Auto-Generation Activity</div>
            {autoGenLog.map((msg,i)=><div key={i} style={{color:"#4a4e6a",marginBottom:2}}>{msg}</div>)}
          </div>}

          <div style={{background:"rgba(52,211,153,.04)",border:"1px solid rgba(52,211,153,.18)",borderRadius:10,padding:14,marginBottom:14}}>
            <SL style={{color:"#34d399"}}>Auto-Generate and Tournament</SL>
            <p style={{fontSize:11,color:"#4a4e6a",marginBottom:12,lineHeight:1.7}}>
              APE <b style={{color:"#34d399"}}>auto-generates</b> every <b style={{color:"#34d399"}}>2 new rows</b> and <b style={{color:"#f87171"}}>auto-prunes the worst performer</b> after every Learn session. The algo pool continuously evolves. Manual buttons for forced runs.
            </p>
            <div style={{display:"flex",gap:8,flexWrap:"wrap",alignItems:"center"}}>
              <PB onClick={doGenerate} style={{background:"#34d399",color:"#060709"}}>🤖 Generate</PB>
              <PB onClick={doTournament} style={{background:"#fbbf24",color:"#060709"}}>🏆 Tournament #{(S.tourN||0)+1}</PB>
              <span style={{fontSize:10,color:"#4a4e6a"}}>{rows.length} rows · {customs.filter(a=>a.generated).length} generated · run #{S.genN||0}</span>
            </div>
            {genMsg&&<div style={{marginTop:10,fontSize:11,color:"#34d399",background:"rgba(52,211,153,.06)",padding:"6px 10px",borderRadius:6}}>{genMsg}</div>}
          </div>
          <Card style={{marginBottom:14}}>
            <SL>Add Custom Algorithm</SL>
            <p style={{fontSize:10,color:"#2d3158",marginBottom:10,lineHeight:1.7}}>Write: <code style={{color:"#a78bfa",background:"#1a1e35",padding:"1px 5px",borderRadius:3}}>(s,M) =&gt; [result]</code> where s is the series array and result is 0–99.</p>
            <div style={{display:"flex",gap:8,marginBottom:8,flexWrap:"wrap"}}>
              <input type="text" value={cName} onChange={e=>setCName(e.target.value)} placeholder="AlgoName" style={{flex:"0 0 130px",background:"#060709",border:"1px solid #1a1e35",color:"#c8d0e8",padding:"7px 10px",borderRadius:6,fontSize:12,fontFamily:"monospace",outline:"none"}}/>
              <input type="text" value={cCode} onChange={e=>setCCode(e.target.value)} placeholder="(s,M) => [M.mod(s[s.length-1] + 7)]" style={{flex:1,minWidth:180,background:"#060709",border:"1px solid #1a1e35",color:"#a78bfa",padding:"7px 10px",borderRadius:6,fontSize:12,fontFamily:"monospace",outline:"none"}}/>
              <PB onClick={addCustom}>Save</PB>
            </div>
            {cErr&&<div style={{fontSize:10,color:"#f87171",marginBottom:6}}>{cErr}</div>}
          </Card>
          {customs.length>0&&<Card style={{marginBottom:14}}>
            <SL>Stored Algorithms ({customs.length})</SL>
            <div style={{display:"flex",flexDirection:"column",gap:6}}>
              {customs.map(algo=><div key={algo.name} style={{background:algo.enabled?"rgba(167,139,250,.04)":"rgba(255,255,255,.01)",border:"1px solid "+(algo.enabled?"rgba(167,139,250,.2)":"#1a1e35"),borderRadius:7,padding:"8px 12px",display:"flex",gap:10,alignItems:"center",flexWrap:"wrap"}}>
                <div style={{flex:1,minWidth:140}}>
                  <div style={{display:"flex",alignItems:"center",gap:6,marginBottom:2}}>
                    <span style={{fontSize:11,fontWeight:700,color:algo.generated?"#34d399":"#a78bfa"}}>{algo.name}</span>
                    <span style={{fontSize:9,padding:"1px 5px",borderRadius:99,background:algo.generated?"rgba(52,211,153,.1)":"rgba(167,139,250,.1)",color:algo.generated?"#34d399":"#a78bfa"}}>{algo.generated?"auto":"custom"}</span>
                  </div>
                  <div style={{fontSize:9,color:"#2d3158",marginBottom:2}}>{algo.desc}</div>
                  <code style={{fontSize:9,color:"#4a4e6a",wordBreak:"break-all"}}>{(algo.code||"").slice(0,80)}{(algo.code||"").length>80?"…":""}</code>
                </div>
                <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
                  <button onClick={()=>{
                    const text="Name: "+algo.name+"\nCode: "+algo.code+"\nDesc: "+algo.desc;
                    navigator.clipboard.writeText(text).catch(()=>{});
                    setCopyMsg(algo.name);setTimeout(()=>setCopyMsg(""),2000);
                  }} style={{background:"rgba(251,191,36,.08)",border:"1px solid rgba(251,191,36,.25)",color:"#fbbf24",padding:"4px 8px",borderRadius:5,cursor:"pointer",fontSize:9,fontFamily:"inherit"}}>
                    {copyMsg===algo.name?"✓ Copied":"📋"}
                  </button>
                  <button onClick={()=>{setCName(algo.name+"_copy");setCCode(algo.code);setTab("algos");}} style={{background:"rgba(52,211,153,.08)",border:"1px solid rgba(52,211,153,.2)",color:"#34d399",padding:"4px 8px",borderRadius:5,cursor:"pointer",fontSize:9,fontFamily:"inherit"}} title="Clone into editor">✎</button>
                  <button onClick={()=>toggleCustom(algo.name)} style={{background:algo.enabled?"rgba(52,211,153,.1)":"rgba(255,255,255,.04)",border:"1px solid "+(algo.enabled?"rgba(52,211,153,.3)":"#1a1e35"),color:algo.enabled?"#34d399":"#4a4e6a",padding:"4px 10px",borderRadius:5,cursor:"pointer",fontSize:9,fontFamily:"inherit"}}>{algo.enabled?"ON":"OFF"}</button>
                  <button onClick={()=>deleteCustom(algo.name)} style={{background:"rgba(248,113,113,.06)",border:"1px solid rgba(248,113,113,.2)",color:"#f87171",padding:"4px 8px",borderRadius:5,cursor:"pointer",fontSize:9,fontFamily:"inherit"}}>✕</button>
                </div>
              </div>)}
            </div>
          </Card>}
          <Card>
            <SL>Intelligence Architecture</SL>
            <div style={{fontSize:11,color:"#4a4e6a",lineHeight:2.1}}>
              <b style={{color:"#c8d0e8"}}>Vote Power =</b> (0.15 + bt×4.0) × global_w × per_row_w × per_range_w × regime_mult<br/>
              <b style={{color:"#c8d0e8"}}>Regime:</b> volatile boosts stat algos ×1.4, flat boosts memory algos ×1.5<br/>
              <b style={{color:"#c8d0e8"}}>Bayesian Momentum:</b> 0.7×prev_mom + 0.3×reward, not raw multipliers<br/>
              <b style={{color:"#c8d0e8"}}>Weight Decay:</b> ×0.97 toward 1.0 each session, prevents staleness<br/>
              <b style={{color:"#c8d0e8"}}>Per-Row:</b> day 15 vs day 28 have separate weight tables<br/>
              <b style={{color:"#c8d0e8"}}>Per-Range:</b> values 0–24, 25–49, 50–74, 75–99 each learn separately<br/>
              <b style={{color:"#c8d0e8"}}>Band:</b> 10th–90th percentile of all algo predictions shown per column<br/>
              <b style={{color:"#c8d0e8"}}>Tournament:</b> bottom 50% custom algos replaced by mutants of top 50%<br/>
              <b style={{color:"#c8d0e8"}}>Phase Space NN:</b> finds nearest historical triplet, returns what followed<br/>
              <b style={{color:"#c8d0e8"}}>Storage:</b> everything auto-saved persistently, never lost<br/>
              <b style={{color:"#c8d0e8"}}>Pseudo-RNG Algos:</b> Xorshift, LFSR7, Rule30, WichmannHill, BBS, MersenneMod, LagFib, ParkMiller, QuadCong, MiddleSquare — detect hidden cyclic structure<br/>
              <b style={{color:"#c8d0e8"}}>PatternMemBank:</b> k-NN on (last 3 values) context window — finds nearest historical match<br/>
              <b style={{color:"#c8d0e8"}}>ModSearch:</b> brute-forces best (v mod k) + offset pattern<br/>
              <b style={{color:"#c8d0e8"}}>XorChain:</b> multi-step XOR across lag positions<br/>
              <b style={{color:"#c8d0e8"}}>PolyCong:</b> quadratic congruential fitting (ax²+bx+c)<br/>
              <b style={{color:"#c8d0e8"}}>Weight Export:</b> save all training progress to file, import on any device — no retraining ever needed<br/>
              <b style={{color:"#c8d0e8"}}>DeepMarkov4:</b> 4-gram context window — finds what follows exact 4-value sequences<br/>
              <b style={{color:"#c8d0e8"}}>GapMarkov:</b> Markov chain on DIFFERENCES — gaps often repeat even when values don't<br/>
              <b style={{color:"#c8d0e8"}}>EntropyAdapt:</b> measures series entropy, auto-switches between frequency and linear<br/>
              <b style={{color:"#c8d0e8"}}>FreqMomentum:</b> finds values rising in frequency — predicts what's trending up<br/>
              <b style={{color:"#c8d0e8"}}>SequenceHash:</b> exact 4-value sequence lookup in history<br/>
              <b style={{color:"#c8d0e8"}}>ValueCluster:</b> k=4 cluster Markov — which cluster follows which<br/>
              <b style={{color:"#c8d0e8"}}>ConsensusFilter:</b> values agreed by 4+ algos get 15% vote boost per extra agreement<br/>
              <b style={{color:"#c8d0e8"}}>HistFreqFilter:</b> values never seen in training get 60% vote penalty<br/>
              <b style={{color:"#c8d0e8"}}>Neural Scores:</b> EMA running accuracy per algo — good algos compound, bad ones fade<br/>
              <b style={{color:"#c8d0e8"}}>Calibration:</b> tracks HIGH/MED/LOW confidence actual accuracy over sessions<br/>
              <b style={{color:"#c8d0e8"}}>Auto-Generate:</b> triggers every 2 new rows automatically — continuous pattern discovery<br/>
              <b style={{color:"#c8d0e8"}}>Auto-Prune:</b> after every Learn session the single worst-scoring generated algo is removed (scored by neural EMA + BT)<br/>
              <b style={{color:"#c8d0e8"}}>User algos:</b> never auto-pruned — only generated algos are pruned<br/>
              <b style={{color:"#c8d0e8"}}>Activity Log:</b> last 5 auto-gen/prune events shown in Algos tab<br/>
              <b style={{color:"#c8d0e8"}}>PRNG Algos:</b> all 10 pseudo-random algos now brute-force fit best parameters to YOUR data — Xorshift shifts, LCG multipliers, LFSR taps, Rule variants, semi-prime moduli all tuned per series
            </div>
          </Card>
        </div>}

      </div>

      {/* ── LIVE TERMINAL PANEL ── */}
      <div style={{position:"fixed",bottom:0,left:0,right:0,zIndex:200,background:"#060709",borderTop:"2px solid #1a1e35"}}>
        {/* Terminal header */}
        <div style={{display:"flex",alignItems:"center",padding:"3px 12px",background:"#0c0e1a",borderBottom:"1px solid #12152a",gap:8}}>
          <div style={{width:7,height:7,borderRadius:"50%",flexShrink:0,background:stClr,boxShadow:"0 0 "+(msg.c==="busy"?8:5)+"px "+stClr}}/>
          <span style={{color:stClr,fontSize:10,flex:1}}>{msg.t}</span>
          <span style={{color:"#1e2240",fontSize:9}}>{rows.length} rows · {accLog.length} sessions · {ALGO_COUNT+customs.length} algos</span>
          <button onClick={()=>setShowTerminal(p=>!p)} style={{background:"transparent",border:"1px solid #1a1e35",color:"#4a4e6a",padding:"1px 8px",borderRadius:4,cursor:"pointer",fontSize:9,fontFamily:"monospace"}}>{showTerminal?"▼ hide":"▲ terminal"}</button>
          <button onClick={()=>setSysLog([{t:Date.now(),lvl:"info",msg:"Terminal cleared"}])} style={{background:"transparent",border:"1px solid #1a1e35",color:"#252840",padding:"1px 8px",borderRadius:4,cursor:"pointer",fontSize:9,fontFamily:"monospace"}}>⊘</button>
        </div>
        {/* Terminal body */}
        {showTerminal&&<div ref={sysRef} style={{height:120,overflowY:"auto",padding:"6px 12px",fontFamily:"'Courier New',monospace",fontSize:10,lineHeight:1.6}}>
          {sysLog.map((e,i)=>{
            const clr=e.lvl==="alert"?"#f87171":e.lvl==="gen"?"#a78bfa":e.lvl==="prune"?"#fbbf24":e.lvl==="learn"?"#34d399":e.lvl==="data"?"#34d399":e.lvl==="refit"?"#60a5fa":"#4a4e6a";
            const ts=new Date(e.t);
            const timeStr=String(ts.getHours()).padStart(2,"0")+":"+String(ts.getMinutes()).padStart(2,"0")+":"+String(ts.getSeconds()).padStart(2,"0");
            return <div key={i} style={{display:"flex",gap:8,marginBottom:1}}>
              <span style={{color:"#252840",flexShrink:0}}>{timeStr}</span>
              <span style={{color:clr}}>{e.msg}</span>
              {e.lvl==="alert"&&<span style={{color:"#f87171",animation:"pulse 1s infinite"}}>◉ SAVE NOW</span>}
            </div>;
          })}
          {sysLog.length===0&&<span style={{color:"#1e2240"}}>No events yet…</span>}
        </div>}
      </div>
    </div>
  );
}

function PredCell(p){
  const pred=p.preds[p.col];
  const item=pred&&pred.top5[p.rank]?pred.top5[p.rank]:null;
  const clr=CLR[p.col];
  return <td style={{padding:"6px 12px",textAlign:"center"}}>
    {item?<span style={{fontWeight:p.rank===0?900:600,color:p.rank===0?clr:"#4a4e6a",fontSize:p.rank===0?17:12}}>
      {pad2(item.value)}{p.rank===0&&<span style={{fontSize:9,color:"#2d3158",marginLeft:3}}>{item.pct}%</span>}
    </span>:<span style={{color:"#1a1e35"}}>—</span>}
  </td>;
}
function ActInput(p){
  const isXX=p.val.toUpperCase()==="XX"||p.val==="";
  const borderClr=isXX?"#2d3158":CLR[p.col]+"44";
  const predVal=p.pred&&p.pred.top5[0]?pad2(p.pred.top5[0].value):"?";
  return <div>
    <div style={{fontSize:9,color:isXX?"#2d3158":CLR[p.col],marginBottom:3,letterSpacing:2}}>Actual {p.col}{isXX?" (XX)":""}</div>
    <div style={{fontSize:9,color:"#2d3158",marginBottom:3}}>Pred: <b style={{color:CLR[p.col]}}>{predVal}</b></div>
    <input type="text" inputMode="numeric" maxLength={2} value={p.val}
      onChange={e=>p.onChg(p.col,e.target.value)}
      placeholder="XX"
      style={{width:54,background:isXX?"#0c0e1a":"#060709",border:"1px solid "+borderClr,color:isXX?"#2d3158":CLR[p.col],padding:"8px",borderRadius:6,fontSize:15,fontFamily:"monospace",textAlign:"center",outline:"none",fontWeight:700,opacity:isXX?0.5:1}}/>
  </div>;
}
function MissingColPred(p){
  const {col,pred,maxV}=p;
  const clr=CLR[col];
  return <div style={{background:"#0c0e1a",border:"1px solid "+clr+"33",borderTop:"2px solid "+clr,borderRadius:8,padding:10}}>
    <div style={{fontSize:12,fontWeight:900,color:clr,marginBottom:6}}>Col {col}</div>
    <div style={{fontSize:9,color:"#2d3158",marginBottom:6}}>Confidence: <span style={{color:pred.confClr,fontWeight:700}}>{pred.conf}</span> · {pred.algoCount} algos</div>
    {pred.top5.map((p2,i)=><div key={i} style={{display:"flex",alignItems:"center",gap:5,marginBottom:i<4?4:0}}>
      <span style={{fontWeight:700,fontSize:i===0?16:10,color:i===0?clr:"#4a4e6a",minWidth:i===0?30:22,fontFamily:"monospace"}}>{i===0?"▶":" "+(i+1)} {pad2(p2.value)}</span>
      <div style={{flex:1,height:i===0?4:2,background:"#1a1e35",borderRadius:99,overflow:"hidden"}}>
        <div style={{height:"100%",width:Math.round(p2.votes/maxV*100)+"%",background:i===0?clr:"#1a1e35",borderRadius:99}}/>
      </div>
      <span style={{fontSize:9,color:"#2d3158",minWidth:22,textAlign:"right"}}>{p2.pct}%</span>
    </div>)}
  </div>;
}
function CheckCard(p){
  const r=p.res;
  if(!r)return null;
  if(r.skipped){
    return <div style={{background:"rgba(37,40,64,.5)",border:"1px solid #1a1e35",borderRadius:8,padding:"8px 12px",textAlign:"center",minWidth:78,opacity:0.5}}>
      <div style={{fontSize:18,marginBottom:3}}>⬜</div>
      <div style={{fontSize:9,color:"#4a4e6a",marginBottom:2}}>Col {p.col}</div>
      <div style={{fontSize:11,color:"#2d3158"}}>XX</div>
      <div style={{fontSize:9,color:"#2d3158",marginTop:2}}>SKIPPED</div>
    </div>;
  }
  const clr=r.exact?"#34d399":r.near?"#fbbf24":"#f87171";
  return <div style={{background:clr+"12",border:"1px solid "+clr+"33",borderRadius:8,padding:"8px 12px",textAlign:"center",minWidth:78}}>
    <div style={{fontSize:18,marginBottom:3}}>{r.exact?"✅":r.near?"🟡":"❌"}</div>
    <div style={{fontSize:9,color:"#4a4e6a",marginBottom:2}}>Col {p.col}</div>
    <div style={{fontSize:11}}><span style={{color:clr,fontWeight:700}}>{pad2(r.predicted||0)}</span>→<span style={{color:"#c8d0e8",fontWeight:700}}>{pad2(r.actual!=null?r.actual:0)}</span></div>
    <div style={{fontSize:9,color:clr,marginTop:2,fontWeight:700}}>{r.exact?"EXACT":r.near?"NEAR ±2":"MISS"}</div>
  </div>;
}
function HeatCell(p){
  const v=p.v;
  const intensity=ok(v)?v/99:0;
  const rgb=p.col==="A"?"167,139,250":p.col==="B"?"52,211,153":p.col==="C"?"251,191,36":"248,113,113";
  const bg=ok(v)?"rgba("+rgb+","+(0.06+intensity*0.55)+")":"#0c0e1a";
  return <td style={{padding:"4px 8px",textAlign:"center",background:bg,color:ok(v)?"#c8d0e8":"#1a1e35",fontWeight:700,fontSize:11}}>{ok(v)?pad2(v):"—"}</td>;
}
function OutlierCol(p){
  const series=getSeries(p.col,p.rows);
  const avg=M.mean(series),std=M.std(series);
  const outliers=p.rows.filter(r=>ok(r[p.col])&&Math.abs(r[p.col]-avg)>2*std);
  return <div>
    <div style={{fontSize:9,color:CLR[p.col],letterSpacing:2,marginBottom:6}}>{p.col} avg:{Math.round(avg)} σ:{std.toFixed(1)}</div>
    {outliers.length?outliers.map(r=><div key={r.row} style={{fontSize:10,color:"#fbbf24",marginBottom:2}}>Row {pad2(r.row)}: <b>{pad2(r[p.col])}</b> ({((r[p.col]-avg)/std).toFixed(1)}σ)</div>):<div style={{fontSize:10,color:"#2d3158"}}>None detected</div>}
  </div>;
}
function HotColdCol(p){
  const series=getSeries(p.col,p.rows);
  const freq={};
  series.forEach(v=>{freq[v]=(freq[v]||0)+1;});
  const sorted=Object.entries(freq).sort((a,b)=>b[1]-a[1]);
  const hot=sorted.slice(0,5),cold=sorted.slice(-3);
  return <div>
    <div style={{fontSize:9,color:CLR[p.col],letterSpacing:2,marginBottom:6}}>{p.col}</div>
    <div style={{marginBottom:4}}>
      <span style={{fontSize:9,color:"#f87171",display:"block",marginBottom:2}}>🔥 Hot</span>
      {hot.map(([v,c])=><span key={v} style={{display:"inline-block",background:"rgba(248,113,113,.1)",border:"1px solid rgba(248,113,113,.2)",borderRadius:4,padding:"2px 5px",fontSize:10,color:"#f87171",margin:"1px",fontWeight:700}}>{pad2(parseInt(v))}<span style={{fontSize:8,color:"#f87171aa"}}>×{c}</span></span>)}
    </div>
    <div>
      <span style={{fontSize:9,color:"#a78bfa",display:"block",marginBottom:2}}>🧊 Cold</span>
      {cold.map(([v,c])=><span key={v} style={{display:"inline-block",background:"rgba(167,139,250,.06)",border:"1px solid rgba(167,139,250,.15)",borderRadius:4,padding:"2px 5px",fontSize:10,color:"#a78bfa",margin:"1px",fontWeight:700}}>{pad2(parseInt(v))}<span style={{fontSize:8,color:"#a78bfaaa"}}>×{c}</span></span>)}
    </div>
  </div>;
}
function CalibCol(p){
  const cal=p.cal;
  const levels=["HIGH","MED","LOW"];
  return <div>
    <div style={{fontSize:9,color:CLR[p.col],letterSpacing:2,marginBottom:6}}>{p.col}</div>
    {levels.map(lv=>{
      const d=cal[lv];
      if(!d||d.total<3)return null;
      const rate=Math.round(d.right/d.total*100);
      const clr=rate>50?"#34d399":rate>25?"#fbbf24":"#f87171";
      return <div key={lv} style={{display:"flex",gap:6,alignItems:"center",marginBottom:4}}>
        <span style={{fontSize:9,color:"#4a4e6a",minWidth:30}}>{lv}</span>
        <div style={{flex:1,height:4,background:"#1a1e35",borderRadius:99}}>
          <div style={{height:"100%",width:rate+"%",background:clr,borderRadius:99}}/>
        </div>
        <span style={{fontSize:9,color:clr,minWidth:32,textAlign:"right"}}>{rate}% ({d.total})</span>
      </div>;
    })}
  </div>;
}
function NeuralScoreCol(p){
  const entries=Object.entries(p.scores).filter(([k])=>!k.startsWith("_")).sort((a,b)=>b[1]-a[1]).slice(0,8);
  const[copied,setCopied]=React.useState("");
  if(!entries.length)return <div style={{fontSize:9,color:"#2d3158"}}>No data yet</div>;
  return <div>
    <div style={{fontSize:9,color:CLR[p.col],letterSpacing:2,marginBottom:6}}>{p.col}</div>
    {entries.map(([name,score])=>{
      const clr=score>1?"#34d399":score>0?"#a78bfa":score<-0.5?"#f87171":"#4a4e6a";
      const customCode=p.customs&&p.customs.find(c=>c.name===name);
      return <div key={name} style={{display:"flex",gap:5,alignItems:"center",marginBottom:3}}>
        <span style={{fontSize:9,color:"#4a4e6a",flex:1,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{name}</span>
        <span style={{fontSize:9,color:clr,fontWeight:700,minWidth:36,textAlign:"right"}}>{score>0?"+":""}{score.toFixed(2)}</span>
        <button onClick={()=>{
          const txt=customCode?"Name: "+name+"\nCode: "+customCode.code:"Algo: "+name+" (built-in)";
          navigator.clipboard.writeText(txt).catch(()=>{});
          setCopied(name);setTimeout(()=>setCopied(""),1500);
        }} style={{background:"transparent",border:"none",color:copied===name?"#34d399":"#2d3158",cursor:"pointer",fontSize:10,padding:"0 2px",lineHeight:1}}>
          {copied===name?"✓":"📋"}
        </button>
      </div>;
    })}
  </div>;
}
function CorrCell(p){
  const v=p.v,abs=Math.abs(v);
  const bg=p.diag?"rgba(167,139,250,.1)":abs>0.7?"rgba(52,211,153,.15)":abs>0.4?"rgba(251,191,36,.1)":"transparent";
  const clr=p.diag?"#a78bfa":v>0.4?"#34d399":v<-0.4?"#f87171":"#4a4e6a";
  return <td style={{padding:"6px 12px",textAlign:"center",background:bg,color:clr,fontWeight:abs>0.4?700:400,fontSize:11}}>{p.diag?"—":v.toFixed(2)}</td>;
}
function CorrTable(p){
  const m=p.corrM;
  return <div>
    <table style={{borderCollapse:"collapse",fontSize:12}}>
      <thead><tr>
        <th style={{padding:"6px 12px",color:"#252840",fontSize:9}}></th>
        {COLS.map(c=><th key={c} style={{padding:"6px 12px",color:CLR[c],fontSize:12,textAlign:"center",fontWeight:900}}>{c}</th>)}
      </tr></thead>
      <tbody>{COLS.map(c1=><tr key={c1}>
        <td style={{padding:"6px 12px",color:CLR[c1],fontWeight:900,fontSize:12}}>{c1}</td>
        {COLS.map(c2=><CorrCell key={c2} v={m[c1]&&m[c1][c2]!=null?m[c1][c2]:0} diag={c1===c2}/>)}
      </tr>)}</tbody>
    </table>
    <div style={{fontSize:10,color:"#2d3158",marginTop:8}}>
      <span style={{color:"#34d399"}}>■</span> strong positive <span style={{color:"#f87171"}}>■</span> strong negative
    </div>
  </div>;
}
const _DAYS=["Sun","Mon","Tue","Wed","Thu","Fri","Sat"];
const _SEASONS=["Winter","Spring","Summer","Fall"];
const _PHASES=["🌑 New","🌓 Waxing","🌕 Full","🌗 Waning"];
function DateBadge({pd,showDate,fullRow}){
  if(!pd)return null;
  return <div style={{marginTop:fullRow?6:8,display:"flex",flexDirection:fullRow?"column":"row",gap:fullRow?3:6,flexWrap:"wrap"}}>
    {showDate&&<div style={{fontSize:10,color:"#a78bfa",fontFamily:"monospace"}}>{showDate}</div>}
    <div style={{display:"flex",gap:4,flexWrap:"wrap",alignItems:"center"}}>
      <span style={{fontSize:8,background:"rgba(167,139,250,.1)",border:"1px solid rgba(167,139,250,.2)",borderRadius:3,padding:"1px 5px",color:"#a78bfa"}}>{_DAYS[pd.dow]}</span>
      <span style={{fontSize:8,background:"rgba(52,211,153,.06)",border:"1px solid rgba(52,211,153,.15)",borderRadius:3,padding:"1px 5px",color:"#34d399"}}>{_SEASONS[pd.season]}</span>
      <span style={{fontSize:8,background:"rgba(251,191,36,.06)",border:"1px solid rgba(251,191,36,.15)",borderRadius:3,padding:"1px 5px",color:"#fbbf24"}}>{_PHASES[pd.lunarPhase]}</span>
      <span style={{fontSize:8,color:"#2d3158"}}>WK{pd.wom} · DR={pd.dateDr}{!fullRow?" · DS="+pd.dateDs:""}</span>
    </div>
  </div>;
}
function Card(p){return <div style={{background:"#0c0e1a",border:"1px solid #1a1e35",borderRadius:10,padding:14,marginBottom:12,...(p.style||{})}}>{p.children}</div>;}
function SL(p){return <div style={{fontSize:9,letterSpacing:3,color:"#2d3158",textTransform:"uppercase",marginBottom:10,...(p.style||{})}}>{p.children}</div>;}
function FI(p){return <div><div style={{fontSize:9,color:p.color||"#4a4e6a",marginBottom:3,letterSpacing:2}}>{p.label}</div><input type="text" inputMode="numeric" maxLength={p.maxLen||2} value={p.val} onChange={e=>p.onChange(e.target.value.replace(/\D/g,"").slice(0,p.maxLen||2))} placeholder={p.maxLen&&p.maxLen>2?"001":"00"} style={{width:p.w||62,background:"#060709",border:"1px solid #1a1e35",color:"#c8d0e8",padding:"8px 6px",borderRadius:6,fontSize:14,fontFamily:"monospace",textAlign:"center",outline:"none"}}/></div>;}
function PB(p){return <button onClick={p.onClick} style={{background:"#a78bfa",border:"none",color:"#fff",padding:"9px 18px",borderRadius:6,cursor:"pointer",fontSize:12,fontFamily:"'Courier New',monospace",fontWeight:700,alignSelf:"flex-end",...(p.style||{})}}>{p.children}</button>;}
function GB(p){return <button onClick={p.onClick} style={{background:"transparent",border:"1px solid #1a1e35",color:"#8892b0",padding:"7px 12px",borderRadius:6,cursor:"pointer",fontSize:11,fontFamily:"'Courier New',monospace",alignSelf:"flex-end"}}>{p.children}</button>;}
