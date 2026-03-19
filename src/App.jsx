import React, { useState, useEffect } from "react";

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
  std:a=>{if(a.length<2)return 0;const avg=M.mean(a);return Math.sqrt(a.reduce((s,v)=>s+(v-avg)**2,0)/a.length);},
  median:a=>{const s=[...a].sort((x,y)=>x-y),m=Math.floor(s.length/2);return s.length%2?s[m]:(s[m-1]+s[m])/2;},
};
const pad2=n=>String(M.mod(n)).padStart(2,"0");
const COLS=["A","B","C","D"];
const ok=v=>v!==null&&v!==undefined&&!isNaN(v);
const getSeries=(col,data)=>data.map(r=>r[col]).filter(v=>ok(v));
const CLR={A:"#a78bfa",B:"#34d399",C:"#fbbf24",D:"#f87171"};

// ── 55 ALGORITHMS ──────────────────────────────
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
  TriNum:         s=>{const v=s[s.length-1]%14;return[M.mod(v*(v+1)/2)];},
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
  ExpSmooth:      s=>{if(s.length<2)return[s[0]||0];let sm=s[0];s.slice(1).forEach(v=>{sm=0.3*v+0.7*sm;});return[M.mod(Math.round(sm))];},
  DblExp:         s=>{if(s.length<3)return[s[s.length-1]];let lv=s[0],tr=s[1]-s[0];for(let i=1;i<s.length;i++){const pl=lv,pt=tr;lv=0.4*s[i]+0.6*(pl+pt);tr=0.3*(lv-pl)+0.7*pt;}return[M.mod(Math.round(lv+tr))];},
  KernelSmooth:   s=>{if(s.length<3)return[s[s.length-1]];const n=s.length,h=3;let ws=0,vs=0;for(let i=0;i<n;i++){const w=Math.exp(-0.5*((n-1-i)/h)**2);ws+=w;vs+=w*s[i];}return[M.mod(Math.round(vs/ws))];},
  MedianFilt:     s=>{if(s.length<3)return[s[s.length-1]];return[M.mod(Math.round(M.median(s.slice(-3))))];},
  LowPass:        s=>{if(s.length<2)return[s[0]||0];let sm=s[0];s.slice(1).forEach(v=>{sm=0.25*v+0.75*sm;});return[M.mod(Math.round(0.25*s[s.length-1]+0.75*sm))];},
  BandPass:       s=>{if(s.length<4)return[s[s.length-1]];const avg=M.mean(s.slice(-8)),std=M.std(s.slice(-8));const filt=s.filter(v=>Math.abs(v-avg)<=std);if(!filt.length)return[s[s.length-1]];return[M.mod(Math.round(M.mean(filt.slice(-4))))];},
  DiffFilt:       s=>{if(s.length<3)return[s[s.length-1]];const diffs=[];for(let i=1;i<s.length;i++){let d=s[i]-s[i-1];if(d>50)d-=100;if(d<-50)d+=100;diffs.push(d);}return[M.mod(s[s.length-1]+Math.round(M.mean(diffs.slice(-4))))];},
  AutoCorr:       s=>{if(s.length<6)return[s[s.length-1]];const n=s.length,avg=M.mean(s);let bestLag=1,bestAcf=-2;for(let lag=1;lag<=Math.min(6,n-2);lag++){let num=0,den=0;for(let i=lag;i<n;i++){num+=(s[i]-avg)*(s[i-lag]-avg);den+=(s[i]-avg)**2;}const acf=den?num/den:0;if(Math.abs(acf)>Math.abs(bestAcf)){bestAcf=acf;bestLag=lag;}}return[M.mod(s[n-bestLag])];},
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
    // Extended search including near-miss scoring
    for(const a of[1,2,3,5,7,11,13,17,21,31,41,51,71,91])
      for(const c of[0,1,3,7,11,13,17,23,29,31,37,43,53,61,71,83,97]){
        let sc=0;
        for(let i=1;i<n;i++){
          const pred=M.mod(a*s[i-1]+c);
          if(pred===s[i])sc+=1;
          else if(M.near(pred,s[i],1))sc+=0.5;
        }
        if(sc>best.sc)best={sc,a,c};
      }
    return[M.mod(best.a*s[n-1]+best.c)];
  },
  Recurrence2:    s=>{
    if(s.length<3)return[s[s.length-1]];
    const n=s.length;let best={sc:-1,pred:s[n-1]};
    for(const a of[1,2,3,-1,-2,5,-3])
      for(const b of[0,1,-1,2,-2,3])
        for(const c of[0,1,-1,3,-3,7,-7,11,-11,13,-13]){
          let sc=0;
          for(let i=2;i<n;i++){
            const p=M.mod(a*s[i-1]+b*s[i-2]+c);
            if(p===s[i])sc+=1;
            else if(M.near(p,s[i],1))sc+=0.4;
          }
          if(sc>best.sc)best={sc,pred:M.mod(a*s[n-1]+b*s[n-2]+c)};
        }
    return[best.pred];
  },
  Cyclic:         s=>{
    if(s.length<4)return[s[s.length-1]];
    const n=s.length;let best={score:-1,pred:s[n-1],period:1};
    for(let p=2;p<=Math.min(12,Math.floor(n/2));p++){
      // Score = sum of (1 - cd/50) for each comparison, gives partial credit
      let sc=0;
      for(let i=p;i<n;i++)sc+=Math.max(0,1-M.cd(s[i],s[i-p])/12);
      const norm=sc/(n-p);
      if(norm>best.score){
        const back=n%p||p;
        best={score:norm,pred:M.mod(s[n-back]||s[n-1]),period:p};
      }
    }
    return[best.pred];
  },
  AR3:            s=>{
    if(s.length<4)return[s[s.length-1]];
    const n=s.length;let best={sc:-1,pred:s[n-1]};
    for(const a of[0.2,0.4,0.5,0.6,0.7,0.8,1.0,1.2])
      for(const b of[-0.4,-0.2,-0.1,0,0.1,0.2,0.4])
        for(const c of[-0.3,-0.1,0,0.1,0.3]){
          let sc=0;
          for(let i=3;i<n;i++){
            const p=M.mod(Math.round(a*s[i-1]+b*s[i-2]+c*s[i-3]));
            if(p===s[i])sc+=1;
            else if(M.near(p,s[i],2))sc+=0.3;
          }
          if(sc>best.sc)best={sc,pred:M.mod(Math.round(a*s[n-1]+b*s[n-2]+c*s[n-3]))};
        }
    return[best.pred];
  },
  MovReg:         s=>{const sl=s.slice(-6),n=sl.length;let sx=0,sy=0,sxy=0,sx2=0;sl.forEach((v,i)=>{sx+=i;sy+=v;sxy+=i*v;sx2+=i*i;});const D=n*sx2-sx*sx;if(!D)return[sl[n-1]];const a=(n*sxy-sx*sy)/D,b=(sy-a*sx)/n;return[M.mod(Math.round(a*n+b))];},
  LogMap:         s=>{if(s.length<3)return[s[s.length-1]];const x=s[s.length-1]/99;let bestR=3.5,bestSc=-1;for(let r=2.5;r<=3.99;r+=0.1){let sc=0,xr=s[0]/99;for(let i=1;i<s.length;i++){xr=r*xr*(1-xr);if(Math.abs(xr*99-s[i])<5)sc++;}if(sc>bestSc){bestSc=sc;bestR=r;}}return[M.mod(Math.round(bestR*(x)*(1-x)*99))];},
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
    // Recent transitions weighted 2x more than old ones
    for(let i=1;i<s.length;i++){
      const k=s[i-1],w=i>s.length-4?2:1;
      if(!tr[k])tr[k]={};
      tr[k][s[i]]=(tr[k][s[i]]||0)+w;
    }
    const k=s[s.length-1];
    if(!tr[k])return[s[s.length-1]];
    return Object.entries(tr[k]).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([v])=>parseInt(v));
  },
  Bigram:         s=>{if(s.length<3)return[s[s.length-1]];const tr={};for(let i=1;i<s.length-1;i++){const k=s[i-1]+"_"+s[i];if(!tr[k])tr[k]={};tr[k][s[i+1]]=(tr[k][s[i+1]]||0)+1;}const k=s[s.length-2]+"_"+s[s.length-1];if(!tr[k])return[s[s.length-1]];return Object.entries(tr[k]).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([v])=>parseInt(v));},
  Trigram:        s=>{if(s.length<4)return[s[s.length-1]];const tr={};for(let i=2;i<s.length-1;i++){const k=s[i-2]+"_"+s[i-1]+"_"+s[i];if(!tr[k])tr[k]={};tr[k][s[i+1]]=(tr[k][s[i+1]]||0)+1;}const k=s[s.length-3]+"_"+s[s.length-2]+"_"+s[s.length-1];if(!tr[k])return[s[s.length-1]];return Object.entries(tr[k]).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([v])=>parseInt(v));},
  ZigZag:         s=>{if(s.length<4)return[s[s.length-1]];const n=s.length;let zz=0;for(let i=1;i<n-1;i++){const a=s[i]-s[i-1],b=s[i+1]-s[i];if((a>0&&b<0)||(a<0&&b>0))zz++;}if(zz/(n-2)>0.6){let d=s[n-1]-s[n-2];if(d>50)d-=100;if(d<-50)d+=100;return[M.mod(s[n-1]-d)];}return[s[n-1]];},
  Sticky:         s=>{const freq={};s.forEach(v=>{freq[v]=(freq[v]||0)+1;});const top=Object.entries(freq).filter(([,c])=>c>=2).sort((a,b)=>b[1]-a[1]).slice(0,4).map(([v])=>parseInt(v));return top.length?top:[s[s.length-1]];},
  XorHeur:        s=>{if(s.length<2)return[s[0]||0];const l=s[s.length-1],p=s[s.length-2];return[(M.d1(l)^M.d1(p))*10+(M.d2(l)^M.d2(p))];},
  RevLag2:        s=>s.length>=3?[M.rev(s[s.length-3])]:[s[s.length-1]],

  // ── PSEUDO-RANDOM STRUCTURE ALGOS (v13 enhanced) ──
  // All now FITTED to data - best parameters found via brute-force search

  // Xorshift: find best shift params via backtest
  Xorshift:       s=>{
    if(s.length<3)return[M.mod(s[s.length-1]^(s[s.length-1]<<3))];
    const n=s.length;let best={sc:-1,a:3,b:5,c:2};
    for(const a of[1,3,5,7])for(const b of[3,5,7,9])for(const c of[1,2,3]){
      let sc=0;
      for(let i=1;i<n;i++){let x=s[i-1]||1;x^=(x<<a)&0xFF;x^=(x>>b)&0xFF;x^=(x<<c)&0xFF;if(M.mod(Math.abs(x))===s[i])sc++;}
      if(sc>best.sc)best={sc,a,b,c};
    }
    let x=s[n-1]||1;x^=(x<<best.a)&0xFF;x^=(x>>best.b)&0xFF;x^=(x<<best.c)&0xFF;
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
    const rules={30:[4,3,2,1],90:[6,3],110:[6,5,3,2,1],150:[6,5,4,1]};
    const n=s.length;let best={sc:-1,ruleName:"30",activeRules:[4,3,2,1]};
    for(const[rname,active]of Object.entries(rules)){
      let sc=0;
      for(let i=1;i<n;i++){const v=s[i-1];let out=0;for(let b=0;b<7;b++){const l=(v>>(b+1))&1,c=(v>>b)&1,r=b>0?(v>>(b-1))&1:0,rule=(l<<2)|(c<<1)|r;if(active.includes(rule))out|=(1<<b);}if(M.mod(out)===s[i])sc++;}
      if(sc>best.sc)best={sc,ruleName:rname,activeRules:active};
    }
    const v=s[n-1];let out=0;
    for(let b=0;b<7;b++){const l=(v>>(b+1))&1,c=(v>>b)&1,r=b>0?(v>>(b-1))&1:0,rule=(l<<2)|(c<<1)|r;if(best.activeRules.includes(rule))out|=(1<<b);}
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

  // ── PATTERN MEMORY BANK ─────────────────────────
  PatternMemBank: s=>{
    if(s.length<6)return[s[s.length-1]];
    const n=s.length,ctx=[s[n-1],s[n-2],s[n-3]];
    // Collect top-3 matches, weight by recency + distance
    const matches=[];
    for(let i=3;i<n-1;i++){
      const d=M.cd(s[i],ctx[0])+M.cd(s[i-1],ctx[1])+M.cd(s[i-2],ctx[2]);
      if(d<=15)matches.push({dist:d,next:s[i+1],recency:i});
    }
    if(!matches.length){
      let best={dist:Infinity,next:s[n-1]};
      for(let i=3;i<n-1;i++){const d=M.cd(s[i],ctx[0])+M.cd(s[i-1],ctx[1])+M.cd(s[i-2],ctx[2]);if(d<best.dist)best={dist:d,next:s[i+1]};}
      return[best.next];
    }
    // Weighted vote: closer + more recent = stronger
    const votes={};
    matches.forEach(m=>{
      const w=(1/(m.dist+1))*(1+m.recency/n);
      votes[m.next]=(votes[m.next]||0)+w;
    });
    return[parseInt(Object.entries(votes).sort((a,b)=>b[1]-a[1])[0][0])];
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
        tr[k][s[i+1]]=(tr[k][s[i+1]]||0)+1;
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
  SumConstraint:  s=>{
    if(s.length<5)return[s[s.length-1]];
    const avg=M.mean(s);
    const std=M.std(s);
    // predict within 1 std of historical mean
    const lo=Math.max(0,Math.round(avg-std));
    const hi=Math.min(99,Math.round(avg+std));
    const mid=Math.round((lo+hi)/2);
    const last=s[s.length-1];
    if(last<lo)return[M.mod(lo+Math.round((hi-lo)*0.25))];
    if(last>hi)return[M.mod(hi-Math.round((hi-lo)*0.25))];
    let g=last-s[s.length-2];if(g>50)g-=100;if(g<-50)g+=100;
    return[M.mod(last+Math.round(g*0.5))];
  },
};

console.log("Algo count:",Object.keys(A).length);

// ── CROSS-COL ──────────────────────────────────
function getCross(col,data){
  const ci=COLS.indexOf(col),aR=COLS[(ci+1)%4],aL=COLS[(ci+3)%4];
  const last=data[data.length-1],prev=data.length>1?data[data.length-2]:null;
  const ser=getSeries(col,data),res={};
  if(!last||ser.length<2)return res;
  [["A","B"],["C","D"],["B","D"],["A","C"],["A","D"],["B","C"]].forEach(([c1,c2])=>{
    if(ok(last[c1])&&ok(last[c2]))res[c1+c2+"Sum"]=[M.mod(last[c1]+last[c2])];
  });
  if(COLS.every(c=>ok(last[c]))){
    res.RowSum=[M.mod(last.A+last.B+last.C+last.D)];
    res.RowHash=[M.mod((last.A*3+last.B*7+last.C*11+last.D*13)%100)];
  }
  const pR=data.filter(r=>ok(r[col])&&ok(r[aR]));
  if(pR.length>=3){
    const diffs=pR.slice(-8).map(r=>{let d=r[col]-r[aR];if(d>50)d-=100;if(d<-50)d+=100;return d;});
    res.ColDiffProj=[M.mod(ser[ser.length-1]+Math.round(M.mean(diffs)))];
  }
  if(ok(last[aL]))res.RevAdj=[M.rev(last[aL])];
  if(ok(last[aR])){res.DsAdj=[M.mod(ser[ser.length-1]+M.ds(last[aR]))];res.AdjRevSum=[M.mod(M.rev(last[aR])+ser[ser.length-1])];}
  if(ok(last[aR])&&ok(last[aL])){
    res.SubCols=[M.mod(last[aL]-last[aR]),M.mod(last[aR]-last[aL])];
    res.XorCols=[M.mod((M.d1(last[aL])^M.d1(last[aR]))*10+(M.d2(last[aL])^M.d2(last[aR])))];
  }
  if(prev&&ok(prev[col])&&ok(prev[aR])&&ok(last[aR]))res.LagDelta=[M.mod(last[aR]+(prev[col]-prev[aR]))];
  // best correlated column
  let bestCorr=-2,bestCol=null;
  COLS.filter(c=>c!==col).forEach(c=>{
    const sx=getSeries(col,data),sy=getSeries(c,data),n=Math.min(sx.length,sy.length);
    if(n<4)return;
    const ax=M.mean(sx.slice(-n)),ay=M.mean(sy.slice(-n));
    let num=0,dx=0,dy=0;
    for(let i=0;i<n;i++){num+=(sx[sx.length-n+i]-ax)*(sy[sy.length-n+i]-ay);dx+=(sx[sx.length-n+i]-ax)**2;dy+=(sy[sy.length-n+i]-ay)**2;}
    const corr=Math.sqrt(dx*dy)?num/Math.sqrt(dx*dy):0;
    if(Math.abs(corr)>Math.abs(bestCorr)){bestCorr=corr;bestCol=c;}
  });
  if(bestCol&&ok(last[bestCol]))res["Corr_"+bestCol]=[M.mod(last[bestCol])];
  return res;
}

// ── BACKTEST ───────────────────────────────────
function btScore(fn,series){
  const n=series.length;if(n<5)return 0.05;
  const from=Math.max(3,n-14);let score=0,cnt=0;
  for(let i=from;i<n;i++){
    try{
      const h=series.slice(0,i),p=fn(h),a=series[i];
      // Precision penalty: wider prediction = lower credit
      const precisionMult=p.length===1?1.0:p.length===2?0.8:p.length===3?0.65:0.5;
      if(p.some(v=>v===a))score+=1.0*precisionMult;
      else if(p.some(v=>M.near(v,a,2)))score+=0.4*precisionMult;
      // Bonus: exact match on first prediction (highest confidence pick)
      if(p[0]===a)score+=0.25;
      cnt++;
    }catch(e){}
  }
  return cnt?score/cnt:0.05;
}

function walkFwd(fn,series){
  const n=series.length,h=Math.min(5,Math.floor(n*0.25));
  if(n<h+4)return null;
  const train=series.slice(0,-h);let ex=0,nr=0,top1ex=0;
  for(let i=0;i<h;i++){
    try{
      const hist=[...train,...series.slice(n-h,n-h+i)],p=fn(hist),a=series[n-h+i];
      if(p.some(v=>v===a))ex++;
      else if(p.some(v=>M.near(v,a,2)))nr++;
      if(p[0]===a)top1ex++;
    }catch(e){}
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
// Learns which LINEAR COMBINATION of algos works best on YOUR data
function buildMetaModel(accLog,algoNames){
  if(!accLog||accLog.length<5)return null;
  // For each past session: collect algo predictions and actual outcome
  // Fit simple ridge regression: actual ≈ w0*algo0 + w1*algo1 + ...
  // Returns weight vector
  const model={weights:{},intercept:50,trained:accLog.length};
  const counts={},sums={};
  algoNames.forEach(name=>{counts[name]=0;sums[name]=0;});
  accLog.forEach(entry=>{
    COLS.forEach(col=>{
      if(!entry.actuals||!entry.actuals[col])return;
      // We stored preds[col] as top1 only — use as signal
      const pred=entry.preds&&entry.preds[col];
      if(pred==null)return;
      const actual=entry.actuals[col];
      const err=M.cd(pred,actual);
      // Algo that was close gets positive signal
      algoNames.forEach(name=>{
        counts[name]=(counts[name]||0)+1;
        sums[name]=(sums[name]||0)+(err<3?1:err<8?0.5:0);
      });
    });
  });
  algoNames.forEach(name=>{
    model.weights[name]=counts[name]>0?sums[name]/counts[name]:0.5;
  });
  return model;
}
function metaBoost(votes,details,metaModel){
  if(!metaModel)return votes;
  const boosted={...votes};
  Object.entries(details).forEach(([name,info])=>{
    const boost=metaModel.weights[name]||0.5;
    if(boost>0.6&&info.pred!=null){
      const v=M.mod(Math.round(info.pred));
      boosted[v]=(boosted[v]||0)*( 0.7+boost*0.6);
    }
  });
  return boosted;
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
function applyForgetting(weights,accLog){
  if(!accLog||accLog.length<3)return weights;
  const next={...weights};
  const recentNames=new Set();
  accLog.slice(-3).forEach(e=>{
    COLS.forEach(col=>{
      const preds=e.preds;
      if(preds&&preds[col]!=null)recentNames.add(String(preds[col]));
    });
  });
  // Accelerate decay for algos not recently contributing
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
  const r=series.slice(-4),o=series.slice(-8,-4);
  if(M.std(r)>M.std(o.length?o:[r[0]])*1.5)return"volatile";
  const gaps=[];for(let i=1;i<r.length;i++){let g=r[i]-r[i-1];if(g>50)g-=100;if(g<-50)g+=100;gaps.push(Math.abs(g));}
  const avgGap=M.mean(gaps);
  if(avgGap<3)return"flat";
  if(avgGap>20)return"trending";
  return"normal";
}
// Regime-gated algo pool: FULL exclusion not just multipliers
const REGIME_POOLS={
  volatile:new Set(["Mean3","Mean5","WtdMean","Median5","HarmMean","GeoMean","MoveStd","ZScore","ExpSmooth","DblExp","KernelSmooth","MedianFilt","LowPass","BandPass","FreqDecay","Sticky","FreqMomentum","ValueCluster","SumConstraint","EntropyAdapt"]),
  flat:new Set(["Markov","Bigram","Trigram","DeepMarkov4","SequenceHash","PatternMemBank","FreqDecay","Sticky","FreqMomentum","GapMarkov","AutoCorr","LagFib","XorChain","ModSearch"]),
  trending:new Set(["WtdMomentum","SecondDiff","LastGap","GapMedian","TheilSen","LinFit","QuadFit","MovReg","DiffSeriesLin","AR3","DblExp","LCGFit","Recurrence2","Cyclic","LogMap","PhaseNN"]),
  normal:null // null = all algos allowed
};
function algoAllowed(name,regime){
  const pool=REGIME_POOLS[regime];
  if(!pool)return true;
  // Always allow cross-col and custom algos regardless of regime
  return pool.has(name);
}

// ── PREDICT ────────────────────────────────────
function predictCol(col,data,W,customs){
  const series=getSeries(col,data);
  if(series.length<3)return null;
  const gw=W.global||{},rw=W.perRow||{},rnw=W.perRange||{};
  const ns=W.neuralScores||{};  // neural running scores
  const maxRow=data.length?Math.max(...data.map(r=>r.row)):1;
  const predRow=maxRow>=31?1:maxRow+1;
  const curRange=Math.floor((series[series.length-1]||0)/25);
  const regime=getRegime(series);
  // v10: full exclusion — silenced algos cast zero votes
  const regimeMult=name=>1.0; // kept for compatibility, filtering done by algoAllowed
  const votes={},contrib={},details={};
  const cast=(name,val,w)=>{
    const v=M.mod(Math.round(val));
    votes[v]=(votes[v]||0)+w;
    if(!contrib[v])contrib[v]=[];
    if(!contrib[v].includes(name))contrib[v].push(name);
  };
  // built-in
  Object.entries(A).forEach(([name,fn])=>{
    if(!algoAllowed(name,regime))return; // regime gate
    try{
      const bt=btScore(fn,series);
      const lw=gw[name]!=null?gw[name]:1;
      const rowW=rw[predRow]?rw[predRow][name]!=null?rw[predRow][name]:1:1;
      const ranW=rnw[curRange]?rnw[curRange][name]!=null?rnw[curRange][name]:1:1;
      // Neural score boost: positive running score boosts, negative penalises
      const nScore=ns[name]!=null?ns[name]:0;
      const nMult=nScore>1.5?1.6:nScore>0.5?1.3:nScore>0?1.1:nScore<-1?0.5:nScore<-0.3?0.75:1.0;
      // Use sqrt(bt) to prevent single high-bt algo from monopolising
      const btFactor=0.2+Math.sqrt(bt)*3.5;
      const w=btFactor*Math.max(0.05,lw)*Math.max(0.1,rowW)*Math.max(0.1,ranW)*nMult;
      const preds=fn(series);
      preds.forEach((p,i)=>cast(name,p,w/(i*0.6+1)));
      details[name]={pred:preds[0],bt:Math.round(bt*100),lw:+lw.toFixed(2),rw:+rowW.toFixed(2),w:+w.toFixed(2),type:"builtin"};
    }catch(e){}
  });
  // cross-col
  const cross=getCross(col,data);
  Object.entries(cross).forEach(([name,preds])=>{
    const lw=gw[name]!=null?gw[name]:1;
    const w=1.8*Math.max(0.05,lw);
    preds.forEach((p,i)=>cast(name,p,w/(i*0.5+1)));
    details[name]={pred:preds[0],bt:null,lw:+lw.toFixed(2),rw:1,w:+w.toFixed(2),type:"cross"};
  });
  // custom
  (customs||[]).forEach(ca=>{
    if(!ca.enabled||!ca.code)return;
    try{
      const fn=makeCustomFn(ca.code);if(!fn)return;
      const bt=btScore(fn,series);
      const lw=gw[ca.name]!=null?gw[ca.name]:1;
      const w=(0.25+bt*5.0)*Math.max(0.05,lw);
      const preds=fn(series);
      preds.forEach((p,i)=>cast(ca.name,p,w/(i*0.6+1)));
      details[ca.name]={pred:preds[0],bt:Math.round(bt*100),lw:+lw.toFixed(2),rw:1,w:+w.toFixed(2),type:"custom"};
    }catch(e){}
  });
  // Adaptive: redistribute tiny votes (< 1% of max) to reduce noise
  const maxVote=Math.max(...Object.values(votes));
  Object.keys(votes).forEach(v=>{if(votes[v]<maxVote*0.01)delete votes[v];});
  const total=Object.values(votes).reduce((a,b)=>a+b,0)||1;
  const top5=Object.entries(votes).sort((a,b)=>b[1]-a[1]).slice(0,5)
    .map(([v,vt])=>({value:parseInt(v),votes:+vt.toFixed(2),pct:Math.round(vt/total*100),algos:contrib[v]||[]}));
  // ── CONSENSUS FILTER: boost values agreed by 3+ independent algos ──
  const algoAgreement={};
  Object.values(details).forEach(info=>{
    const v=info.pred!=null?M.mod(Math.round(info.pred)):-1;
    if(v>=0)algoAgreement[v]=(algoAgreement[v]||0)+1;
  });
  // Re-boost values with high agreement
  Object.entries(algoAgreement).forEach(([v,cnt])=>{
    if(cnt>=3&&votes[parseInt(v)])votes[parseInt(v)]*=1+(cnt-2)*0.12;
  });

  // ── META-LEARNER BOOST ──────────────────────────
  const metaModel=buildMetaModel(series.length>=10?[]:[],(Object.keys(A)));
  // (meta boost applied inline per algo above via neural scores — full model needs session data)

  // ── JOINT COLUMN HINT ────────────────────────────
  // Will be applied by caller after all 4 cols predicted

  // ── HISTORICAL FREQ FILTER: down-vote values that never appeared ──
  const histValues=new Set(series);
  Object.keys(votes).forEach(v=>{
    // Graduated: values seen once get 0.7x, never seen get 0.35x
  const vInt=parseInt(v);
  const seenCount=series.filter(x=>x===vInt).length;
  if(seenCount===0)votes[vInt]*=0.35;
  else if(seenCount===1)votes[vInt]*=0.7;
  });

  const ac=Object.keys(details).length;
  const consensus=top5[0]?Math.round(top5[0].algos.length/ac*100):0;
  const t1pct=top5[0]?top5[0].pct:0;
  const conf=t1pct>40?"HIGH":t1pct>20?"MED":"LOW";
  const confClr=conf==="HIGH"?"#34d399":conf==="MED"?"#fbbf24":"#f87171";
  const allP=Object.values(details).map(d=>d.pred).filter(ok);
  const sAllP=[...allP].sort((a,b)=>a-b);
  const lo=sAllP[Math.floor(sAllP.length*0.1)]||top5[0]?.value||0;
  const hi=sAllP[Math.floor(sAllP.length*0.9)]||top5[0]?.value||0;
  return{top5,details,consensus,algoCount:ac,conf,confClr,variance:allP.length>1?+M.std(allP).toFixed(1):0,regime,bandLo:lo,bandHi:hi};
}

// ── NEURAL RUNNING SCORE (per-algo accuracy tracker) ──
function updateNeuralScores(pred,actual,prevScores){
  const next={...prevScores};
  if(!pred)return next;
  Object.entries(pred.details).forEach(([name,info])=>{
    if(!ok(info.pred))return;
    const p=M.mod(Math.round(info.pred));
    const ex=p===actual;
    const nr=!ex&&M.near(p,actual,2);
    const reward=ex?3:nr?1:-0.5;
    const prev=next[name]!=null?next[name]:0;
    // exponential moving average: 80% old, 20% new
    next[name]=+(0.8*prev+0.2*reward).toFixed(3);
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

// ── WEIGHT UPDATE ──────────────────────────────
function updateW(pred,actual,W,predRow){
  const gw={...W.global||{}};
  const rw={...W.perRow||{}};
  const rnw={...W.perRange||{}};
  if(!pred)return{global:gw,perRow:rw,perRange:rnw};
  const cr=Math.floor(actual/25);
  if(!rw[predRow])rw[predRow]={};
  if(!rnw[cr])rnw[cr]={};
  Object.entries(pred.details).forEach(([name,info])=>{
    if(!ok(info.pred))return;
    const p=M.mod(Math.round(info.pred));
    const ex=p===actual,nr=!ex&&M.near(p,actual,2);
    const mult=ex?1.4:nr?1.1:0.80;
    const mom=gw["_m_"+name]!=null?gw["_m_"+name]:1.0;
    const newMom=0.75*mom+0.25*mult; // slightly more momentum retention
    gw["_m_"+name]=Math.min(2.0,Math.max(0.3,newMom)); // clamp momentum
    const prev=gw[name]!=null?gw[name]:1;
    // Decay factor depends on how far from 1.0 (pull toward baseline)
    const decay=prev>1?0.96:0.98;
    gw[name]=Math.min(6,Math.max(0.04,prev*newMom))*decay+(1-decay);
    const rp=rw[predRow][name]!=null?rw[predRow][name]:1;
    rw[predRow][name]=Math.min(5,Math.max(0.05,rp*(ex?1.5:nr?1.15:0.75)));
    const rn=rnw[cr][name]!=null?rnw[cr][name]:1;
    rnw[cr][name]=Math.min(5,Math.max(0.05,rn*mult));
  });
  return{global:gw,perRow:rw,perRange:rnw,neuralScores:updateNeuralScores(pred,actual,W.neuralScores||{})};
}

// ── ADAPTIVE ALGO GENERATOR ────────────────────
function generateAlgos(data,existing){
  const out=[];
  COLS.forEach(col=>{
    const s=getSeries(col,data),n=s.length;
    if(n<6)return;
    // best linear
    let bl={sc:-1,a:1,b:0};
    for(let a=1;a<10;a++)for(let b=0;b<100;b+=3){let sc=0;for(let i=1;i<n;i++)if(M.mod(a*s[i-1]+b)===s[i])sc++;if(sc>bl.sc)bl={sc,a,b};}
    if(bl.sc>1){const nm="Lin_"+col+"_"+bl.a+"x"+bl.b;if(!existing.has(nm))out.push({name:nm,code:"(s,M)=>[M.mod("+bl.a+"*s[s.length-1]+"+bl.b+")]",desc:"Auto linear col "+col,enabled:true,generated:true,createdAt:Date.now()});}
    // best cyclic
    let bc={sc:-1,p:2};
    for(let p=2;p<=8;p++){let sc=0;for(let i=p;i<n;i++)if(M.cd(s[i],s[i-p])<=2)sc++;if(sc/(n-p)>bc.sc)bc={sc:sc/(n-p),p};}
    if(bc.sc>0.35){const nm="Cyc_"+col+"_p"+bc.p;if(!existing.has(nm))out.push({name:nm,code:"(s,M)=>{const n=s.length,p="+bc.p+",b=n%p||p;return[M.mod(s[n-b]||s[n-1])];}",desc:"Auto cyclic p"+bc.p+" col "+col,enabled:true,generated:true,createdAt:Date.now()});}
    // avg gap
    const gaps=[];for(let i=1;i<n;i++){let g=s[i]-s[i-1];if(g>50)g-=100;if(g<-50)g+=100;gaps.push(g);}
    const avgGap=Math.round(M.mean(gaps.slice(-6)));
    if(avgGap!==0){const nm="Gap_"+col+"_"+(avgGap>0?"+":"")+avgGap;if(!existing.has(nm))out.push({name:nm,code:"(s,M)=>[M.mod(s[s.length-1]+("+avgGap+"))]",desc:"Auto gap "+(avgGap>0?"+":"")+avgGap+" col "+col,enabled:true,generated:true,createdAt:Date.now()});}
    // crossover: blend top 2 algo predictions using stored offset
    const scored=Object.entries(A).map(([nm,fn])=>({nm,sc:btScore(fn,s)})).sort((a,b)=>b.sc-a.sc);
    if(scored.length>=2&&scored[0].sc>0.15){
      try{
        const p1=A[scored[0].nm](s);
        const p2=A[scored[1].nm](s);
        const blended=M.mod(Math.round((p1[0]+p2[0])/2));
        const offset=((blended-s[s.length-1]+150)%100)-50;
        const nm="Cross_"+col+"_"+scored[0].nm.slice(0,4)+"_"+scored[1].nm.slice(0,4);
        if(!existing.has(nm))out.push({name:nm,code:"(s,M)=>[M.mod(s[s.length-1]+("+offset+"))]",desc:"Auto crossover "+scored[0].nm+"+"+scored[1].nm+" col "+col,enabled:true,generated:true,createdAt:Date.now()});
      }catch(e){}
    }
  });
  return out.slice(0,24);
}

// ── TOURNAMENT ─────────────────────────────────
function runTournament(customs,data){
  const scored=customs.map(ca=>{
    let tot=0;
    COLS.forEach(col=>{const s=getSeries(col,data);if(s.length<5)return;const fn=makeCustomFn(ca.code);if(fn)tot+=btScore(fn,s);});
    return{...ca,_sc:tot/4};
  }).sort((a,b)=>b._sc-a._sc);
  if(scored.length<4)return customs;
  const half=Math.ceil(scored.length/2);
  const top=scored.slice(0,half);
  const mutants=scored.slice(half).map((_,i)=>{
    const parent=top[i%top.length];
    return{...parent,name:"Mut_"+parent.name.slice(-8)+"_"+Date.now()%1000,desc:"Mutant of "+parent.name,generated:true,createdAt:Date.now()};
  });
  return[...top,...mutants];
}

// ── AUTO-GENERATE TRIGGER ──────────────────────
// Returns true if conditions are met to auto-generate
function shouldAutoGenerate(rows,genN,lastAutoGenRows){
  if(rows<4)return false;
  // Auto-generate every 2 new rows
  const rowsSinceLast=(lastAutoGenRows||0);
  return(rows-rowsSinceLast)>=2;
}

// ── AUTO-PRUNE: remove worst generated algo every session ──────
function scoreAlgo(ca,weights,rows){
  if(!ca.generated)return Infinity; // never score user algos
  const fn=makeCustomFn(ca.code);
  if(!fn)return -999; // broken code = worst score
  // Composite score = neural score (60%) + backtest (40%)
  let totalNs=0,nsCount=0,totalBt=0,btCnt=0;
  COLS.forEach(col=>{
    const ns=weights[col]&&weights[col].neuralScores?weights[col].neuralScores:{};
    if(ns[ca.name]!=null){totalNs+=ns[ca.name];nsCount++;}
    const s=getSeries(col,rows);
    if(s.length>=5){totalBt+=btScore(fn,s);btCnt++;}
  });
  const avgNs=nsCount?totalNs/nsCount:0;
  const avgBt=btCnt?totalBt/btCnt:0.05;
  return avgNs*0.6+avgBt*3.0; // combined score
}

function pruneWeakAlgos(customs,weights,rows){
  if(!customs||customs.length===0)return{pruned:customs,removed:[]};
  const generated=customs.filter(ca=>ca.generated);
  const userDefined=customs.filter(ca=>!ca.generated);
  if(generated.length<2)return{pruned:customs,removed:[]}; // keep at least 1
  if(rows.length<4)return{pruned:customs,removed:[]};

  // Score every generated algo
  const scored=generated.map(ca=>({ca,score:scoreAlgo(ca,weights,rows)}));
  scored.sort((a,b)=>a.score-b.score); // lowest score = worst

  const worst=scored[0];
  const removed=[];

  // Always remove the single worst IF it's genuinely bad (score < threshold)
  // Threshold gets stricter as we have more data
  const threshold=rows.length>=20?0.1:rows.length>=10?0.0:-0.5;
  if(worst.score<threshold){
    removed.push({name:worst.ca.name,reason:"score:"+worst.score.toFixed(2)});
    const pruned=[...userDefined,...scored.slice(1).map(x=>x.ca)];
    return{pruned,removed};
  }
  return{pruned:customs,removed:[]};
}

// ── EXPORT HELPERS ─────────────────────────────
function doExportCSV(data,preds,predRow){
  const rows=data.map(r=>pad2(r.row)+","+(r.A!=null?r.A:"XX")+","+(r.B!=null?r.B:"XX")+","+(r.C!=null?r.C:"XX")+","+(r.D!=null?r.D:"XX")).join("\n");
  let pLine="";
  if(preds&&predRow){
    const pa=preds.A?preds.A.top5[0]?.value:null,pb=preds.B?preds.B.top5[0]?.value:null;
    const pc=preds.C?preds.C.top5[0]?.value:null,pd=preds.D?preds.D.top5[0]?.value:null;
    pLine="\n"+pad2(predRow)+","+(ok(pa)?pad2(pa):"?")+","+(ok(pb)?pad2(pb):"?")+","+(ok(pc)?pad2(pc):"?")+","+(ok(pd)?pad2(pd):"?")+" (PRED)";
  }
  const blob=new Blob(["Row,A,B,C,D\n"+rows+pLine],{type:"text/csv"});
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
function parseImportedWeights(parsed){
  // Accept both full state backup and weights-only export
  if(parsed.version==="ape-v13-weights"||parsed.version==="ape-v7-weights"){
    return{weights:parsed.weights,customs:parsed.customs||[],accLog:parsed.accLog||[]};
  }
  // Full state backup
  if(parsed.weights&&parsed.customs!==undefined){
    return{weights:parsed.weights,customs:parsed.customs,accLog:parsed.accLog||[]};
  }
  return null;
}

// ── STORAGE ────────────────────────────────────
const SK="ape-v13";
async function saveS(s){try{localStorage.setItem(SK,JSON.stringify(s));}catch(e){}}
async function loadS(){try{const r=localStorage.getItem(SK);return r?JSON.parse(r):null;}catch(e){return null;}}
function fresh(){
  return{
    datasets:{def:{name:"Dataset 1",rows:[]}},
    active:"def",
    weights:{A:{global:{},perRow:{},perRange:{},neuralScores:{}},B:{global:{},perRow:{},perRange:{},neuralScores:{}},C:{global:{},perRow:{},perRange:{},neuralScores:{}},D:{global:{},perRow:{},perRange:{},neuralScores:{}}},
    calibration:{A:{},B:{},C:{},D:{}},
    customs:[],accLog:[],preds:null,predRow:null,genN:0,tourN:0,lastAutoGenRows:0,pruneLog:[]
  };
}

// ── APP ────────────────────────────────────────
export default function App(){
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
  const[weightsMsg,setWeightsMsg]=useState("");
  const[showCalib,setShowCalib]=useState(false);
  const[autoGenLog,setAutoGenLog]=useState([]);
  const[missingPreds,setMissingPreds]=useState({});

  const st=(t,c)=>setMsg({t,c:c||"ok"});

  function upd(fn){
    setS(prev=>{
      const next=typeof fn==="function"?fn(prev):{...prev,...fn};
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
    loadS().then(saved=>{
      if(saved){
        if(saved.dataset&&!saved.datasets){saved.datasets={def:{name:"Dataset 1",rows:saved.dataset}};saved.active="def";delete saved.dataset;}
        if(!saved.active)saved.active="def";
        if(!saved.weights||!saved.weights.A||!saved.weights.A.global)saved.weights={A:{global:{},perRow:{},perRange:{},neuralScores:{}},B:{global:{},perRow:{},perRange:{},neuralScores:{}},C:{global:{},perRow:{},perRange:{},neuralScores:{}},D:{global:{},perRow:{},perRange:{},neuralScores:{}}};
        COLS.forEach(col=>{if(!saved.weights[col].neuralScores)saved.weights[col].neuralScores={};});
        if(!saved.calibration)saved.calibration={A:{},B:{},C:{},D:{}};
        if(!saved.customs)saved.customs=[];
        if(!saved.genN)saved.genN=0;
        if(!saved.tourN)saved.tourN=0;
        setS(saved);
        const n=saved.datasets&&saved.datasets[saved.active]?saved.datasets[saved.active].rows.length:0;
        st("Restored — "+n+" rows · "+(saved.customs?saved.customs.length:0)+" algos · "+(saved.accLog?saved.accLog.length:0)+" sessions");
      }else{
        st("Welcome to APE v7");
      }
      setLoaded(true);
    });
  },[]);

  function addRow(){
    const r=parseInt(rowIn);
    if(isNaN(r)||r<1||r>31){st("Row must be 1–31","err");return;}
    const entry={row:r};
    for(let i=0;i<COLS.length;i++){
      const col=COLS[i],raw=vals[col].trim();
      if(!raw||raw.toUpperCase()==="XX"){entry[col]=null;continue;}
      const n=parseInt(raw);if(isNaN(n)||n<0||n>99){st(col+" must be 00–99 or XX","err");return;}
      entry[col]=n;
    }
    if(COLS.every(c=>entry[c]===null)){st("Enter at least one value","err");return;}
    setRows(prev=>{
      const updated=[...prev.filter(x=>x.row!==r),entry].sort((a,b)=>a.row-b.row);
      // Auto-generate check after row added
      setTimeout(()=>{
        setS(cur=>{
          const curRows=cur.datasets&&cur.datasets[cur.active]?cur.datasets[cur.active].rows:[];
          if(shouldAutoGenerate(curRows.length,cur.genN,cur.lastAutoGenRows)){
            const existing=new Set([...Object.keys(A),...(cur.customs||[]).map(a=>a.name)]);
            const newAlgos=generateAlgos(curRows,existing);
            if(newAlgos.length>0){
              const log=cur.autoGenLog||[];
              const msg="+"+newAlgos.length+" algos auto-generated ("+curRows.length+" rows)";
              const next={...cur,customs:[...(cur.customs||[]),...newAlgos],genN:(cur.genN||0)+1,lastAutoGenRows:curRows.length,autoGenLog:[...log.slice(-9),{at:new Date().toISOString(),msg}]};
              saveS(next);
              setAutoGenLog(p=>[...p.slice(-4),msg]);
              return next;
            }
          }
          return cur;
        });
      },100);
      return updated;
    });
    setRowIn(String(r+1).padStart(2,"0"));setVals({A:"",B:"",C:"",D:""});
    st("Row "+pad2(r)+" saved ✓");
  }

  function doBulk(){
    const lines=bulk.trim().split("\n").filter(l=>l.trim());let added=0,errs=0;
    const next=[...rows];
    lines.forEach(line=>{
      const pts=line.split(",").map(p=>p.trim());if(pts.length<5){errs++;return;}
      const row=parseInt(pts[0]);if(isNaN(row)||row<1||row>31){errs++;return;}
      const abcd=pts.slice(1,5).map(p=>{if(!p||p.toUpperCase()==="XX")return null;const n=parseInt(p);return(isNaN(n)||n<0||n>99)?undefined:n;});
      if(abcd.some(v=>v===undefined)){errs++;return;}
      const idx=next.findIndex(x=>x.row===row);
      if(idx>=0)next[idx]={row,A:abcd[0],B:abcd[1],C:abcd[2],D:abcd[3]};else next.push({row,A:abcd[0],B:abcd[1],C:abcd[2],D:abcd[3]});
      added++;
    });
    next.sort((a,b)=>a.row-b.row);setRows(()=>next);setBulk("");setShowBulk(false);
    // Auto-generate after bulk import
    setTimeout(()=>{
      setS(cur=>{
        const curRows=cur.datasets&&cur.datasets[cur.active]?cur.datasets[cur.active].rows:[];
        if(shouldAutoGenerate(curRows.length,cur.genN,cur.lastAutoGenRows)){
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
        if(!parsed.weights||!parsed.weights.A||!parsed.weights.A.global)parsed.weights={A:{global:{},perRow:{},perRange:{}},B:{global:{},perRow:{},perRange:{}},C:{global:{},perRow:{},perRange:{}},D:{global:{},perRow:{},perRange:{}}};
        if(!parsed.customs)parsed.customs=[];
        setS(parsed);saveS(parsed);
        st("Imported successfully");
      }catch(err){st("Import failed: "+err.message,"err");}
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
    const target=maxRow>=31?1:maxRow+1;
    const result={};
    COLS.forEach(col=>{result[col]=predictCol(col,rows,S.weights[col],S.customs);});
    // Joint column hint: use predictions of other cols to refine each col
    const knownPreds={};
    COLS.forEach(col=>{if(result[col])knownPreds[col]=result[col].top5[0]?.value;});
    COLS.forEach(col=>{
      if(!result[col])return;
      const hint=jointColHint(col,rows,knownPreds);
      if(hint!=null){
        // Inject joint hint as extra votes for that value
        const top=result[col].top5;
        const existing=top.find(p=>p.value===hint);
        if(existing)existing.pct=Math.min(99,existing.pct+8);
        else if(top.length>0)top[top.length-1]={value:hint,votes:top[top.length-1].votes,pct:Math.min(15,top[top.length-1].pct),algos:["JointCol"]};
      }
    });
    upd(prev=>({...prev,preds:result,predRow:target}));
    setCheckRes(null);setActs({A:"",B:"",C:"",D:""});setTab("predict");
    setTimeout(()=>st("Predictions ready ✓"),300);
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

    missing.forEach(col=>{
      // Run full prediction engine with the partial row included
      const pred=predictCol(col,tempDataset,S.weights[col],S.customs);
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
      COLS.forEach(col=>{
        // Only update weights for columns where we know the actual
        if(actuals[col]===null)return;
        let updated=updateW(prev.preds[col],actuals[col],prev.weights[col],prev.predRow);
        updated.global=applyForgetting(updated.global,prev.accLog);
        nw[col]=updated;
        const pred=prev.preds[col];
        if(pred)nc[col]=updateCalibration(pred.conf,actuals[col]===pred.top5[0]?.value,nc[col]||{});
      });
      const knownCount=knownCols.length;
      const entry={at:new Date().toISOString(),targetRow:prev.predRow,preds:Object.fromEntries(COLS.map(c=>[c,prev.preds[c]&&prev.preds[c].top5[0]?prev.preds[c].top5[0].value:null])),actuals,results,exactCount,knownCount};
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
      const dsKey=prev.active;
      const existingRows=prev.datasets[dsKey]?prev.datasets[dsKey].rows:[];
      const newRows=[...existingRows.filter(r=>r.row!==prev.predRow),autoRow].sort((a,b)=>a.row-b.row);
      const newDs={...prev.datasets,[dsKey]:{...prev.datasets[dsKey],rows:newRows}};
      return{...prev,weights:nw,calibration:nc,customs:pruned,datasets:newDs,accLog:[...(prev.accLog||[]).slice(-99),entry]};
    });
    st("Learned! "+exactCount+"/4 exact — weights saved ✓");
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

  function doTournament(){
    if(!S.customs||S.customs.length<4){st("Need at least 4 custom algos","warn");return;}
    const evolved=runTournament(S.customs,rows);
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
  const totalExact=accLog.reduce((s,e)=>s+e.exactCount,0);
  const overallPct=accLog.length?Math.round(totalExact/(accLog.length*4)*100):0;
  function handleActChange(col,val){
    const v=val.toUpperCase();
    const cleaned=v==="X"||v==="XX"?"XX":v.replace(/\D/g,"").slice(0,2);
    setActs(p=>({...p,[col]:cleaned}));
  }
  const stClr=msg.c==="ok"?"#34d399":msg.c==="err"?"#f87171":msg.c==="warn"?"#fbbf24":msg.c==="busy"?"#a78bfa":"#4a4e6a";
  const dsKeys=Object.keys(S.datasets||{});
  const customs=S.customs||[];

  // Row difficulty
  const rowDifficulty=computeRowDifficulty(accLog);
  const predRowDiff=S.predRow&&rowDifficulty[S.predRow]!=null?rowDifficulty[S.predRow]:null;

  // Streak tracker
  const streaks={};
  COLS.forEach(col=>{
    let s=0;
    for(let i=accLog.length-1;i>=0;i--){
      if(accLog[i].results&&accLog[i].results[col]&&accLog[i].results[col].exact)s++;
      else break;
    }
    streaks[col]=s;
  });

  // Missing rows (computed as plain array, no IIFE)
  const sortedRowNums=rows.map(r=>r.row).sort((a,b)=>a-b);
  const missingRows=[];
  for(let i=1;i<sortedRowNums.length;i++){
    for(let j=sortedRowNums[i-1]+1;j<sortedRowNums[i];j++)missingRows.push(j);
  }

  if(!loaded)return React.createElement("div",{style:{background:"#060709",minHeight:"100vh",display:"flex",alignItems:"center",justifyContent:"center",color:"#4a4e6a",fontFamily:"monospace"}},"Loading…");

  return (
    <div style={{background:"#060709",color:"#c8d0e8",minHeight:"100vh",fontFamily:"'Courier New',monospace",fontSize:13,backgroundImage:"radial-gradient(ellipse at 15% 0%,rgba(124,109,250,.07) 0%,transparent 55%),radial-gradient(ellipse at 85% 100%,rgba(52,211,153,.05) 0%,transparent 55%)"}}>
      <div style={{maxWidth:1100,margin:"0 auto",padding:"14px 12px 70px"}}>

        <div style={{textAlign:"center",padding:"18px 0 10px"}}>
          <div style={{fontSize:9,letterSpacing:5,color:"#252840",marginBottom:5,textTransform:"uppercase"}}>Self-Learning · Adaptive · Prediction · Engine</div>
          <div style={{fontSize:"clamp(28px,6vw,48px)",fontWeight:900,letterSpacing:-2,lineHeight:1,background:"linear-gradient(135deg,#a78bfa,#c4b5fd 35%,#34d399 70%,#6ee7b7)",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent",backgroundClip:"text"}}>APE v13</div>
          <div style={{fontSize:9,color:"#252840",marginTop:4}}>{Object.keys(A).length} built-in · sequential patterns · reverse detect · step/bounce algos · fitted PRNG · auto-gen · auto-prune</div>
          {accLog.length>0&&<div style={{marginTop:8,display:"inline-flex",gap:8,alignItems:"center",background:"rgba(52,211,153,.07)",border:"1px solid rgba(52,211,153,.18)",borderRadius:99,padding:"3px 14px",fontSize:10,color:"#34d399"}}>🧠 {accLog.length} sessions · {overallPct}% exact · {customs.length} algos</div>}
        </div>

        <div style={{display:"flex",gap:6,padding:"6px 0 10px",overflowX:"auto",flexWrap:"wrap",alignItems:"center"}}>
          {dsKeys.map(id=><button key={id} onClick={()=>upd(prev=>({...prev,active:id}))} style={{background:S.active===id?"rgba(167,139,250,.15)":"transparent",border:"1px solid "+(S.active===id?"rgba(167,139,250,.4)":"#1a1e35"),color:S.active===id?"#a78bfa":"#4a4e6a",padding:"3px 10px",borderRadius:99,cursor:"pointer",fontSize:10,fontFamily:"inherit",whiteSpace:"nowrap"}}>{S.datasets[id]?S.datasets[id].name:"?"} ({S.datasets[id]&&S.datasets[id].rows?S.datasets[id].rows.length:0})</button>)}
          <input value={dsName} onChange={e=>setDsName(e.target.value)} placeholder="New dataset…" style={{background:"#060709",border:"1px solid #1a1e35",color:"#c8d0e8",padding:"3px 8px",borderRadius:99,fontSize:10,fontFamily:"monospace",outline:"none",width:100}}/>
          <GB onClick={addDataset}>＋</GB>
        </div>

        <div style={{display:"flex",borderBottom:"1px solid #12152a",marginBottom:14,overflowX:"auto"}}>
          {[{id:"data",l:"📊 Data ("+rows.length+")"},{id:"predict",l:"🔮 Predict"},{id:"learn",l:"✅ Learn"+(accLog.length?" ("+accLog.length+")":"")},{id:"analysis",l:"📈 Analysis"},{id:"algos",l:"⚙ Algos ("+(Object.keys(A).length+customs.length)+")"}].map(function(t){
            return <button key={t.id} onClick={()=>setTab(t.id)} style={{background:tab===t.id?"rgba(167,139,250,.1)":"transparent",border:"none",borderBottom:tab===t.id?"2px solid #a78bfa":"2px solid transparent",color:tab===t.id?"#a78bfa":"#4a4e6a",padding:"8px 12px",cursor:"pointer",fontSize:11,fontFamily:"inherit",whiteSpace:"nowrap",marginBottom:-1}}>{t.l}</button>;
          })}
          <div style={{flex:1}}/>
          <button onClick={runPredict} style={{background:"linear-gradient(135deg,#7c3aed,#a78bfa)",border:"none",color:"#fff",padding:"7px 16px",borderRadius:7,cursor:"pointer",fontSize:11,fontWeight:700,fontFamily:"inherit",alignSelf:"center",marginRight:2}}>🔮 Predict</button>
        </div>

        {tab==="data"&&<div>
          <Card>
            <SL>Add Row — {S.datasets[S.active]?S.datasets[S.active].name:"?"}</SL>
            <div style={{display:"flex",gap:8,flexWrap:"wrap",alignItems:"flex-end"}}>
              <FI label="Row #" val={rowIn} onChange={setRowIn} w={56}/>
              <FI label="A" val={vals.A} color={CLR.A} onChange={v=>setVals(p=>({...p,A:v}))} w={54}/>
              <FI label="B" val={vals.B} color={CLR.B} onChange={v=>setVals(p=>({...p,B:v}))} w={54}/>
              <FI label="C" val={vals.C} color={CLR.C} onChange={v=>setVals(p=>({...p,C:v}))} w={54}/>
              <FI label="D" val={vals.D} color={CLR.D} onChange={v=>setVals(p=>({...p,D:v}))} w={54}/>
              <PB onClick={addRow}>＋ Add</PB>
              <GB onClick={()=>setRowIn(String(new Date().getDate()).padStart(2,"0"))}>📅 Today</GB>
            </div>
            {missingRows.length>0&&<div style={{marginTop:8,fontSize:10,color:"#fbbf24",background:"rgba(251,191,36,.06)",border:"1px solid rgba(251,191,36,.18)",borderRadius:5,padding:"4px 10px"}}>⚠ Missing rows: {missingRows.map(r=>pad2(r)).join(", ")}</div>}
          </Card>
          <div style={{marginBottom:12}}>
            <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
              <GB onClick={()=>setShowBulk(!showBulk)}>{showBulk?"▲":"▼"} Bulk CSV</GB>
              <GB onClick={()=>doExportCSV(rows,S.preds,S.predRow)}>📥 CSV</GB>
              <GB onClick={()=>doExportJSON(S)}>📦 JSON</GB>
              <label style={{background:"transparent",border:"1px solid #1a1e35",color:"#8892b0",padding:"7px 12px",borderRadius:6,cursor:"pointer",fontSize:11,fontFamily:"inherit"}}>📂 Import<input type="file" accept=".json" onChange={doImport} style={{display:"none"}}/></label>
            </div>
            {showBulk&&<div style={{marginTop:8,background:"#0c0e1a",border:"1px solid #1a1e35",borderRadius:8,padding:12}}>
              <textarea value={bulk} onChange={e=>setBulk(e.target.value)} placeholder={"01,02,10,92,XX\n02,91,10,30,68"} style={{width:"100%",height:90,background:"#060709",border:"1px solid #1a1e35",color:"#c8d0e8",padding:8,borderRadius:6,fontSize:11,resize:"vertical",fontFamily:"monospace",outline:"none",boxSizing:"border-box"}}/>
              <div style={{display:"flex",gap:6,marginTop:8}}><PB onClick={doBulk}>Import</PB><GB onClick={()=>setShowBulk(false)}>Cancel</GB></div>
            </div>}
          </div>
          {rows.length>0?<Card>
            <div style={{overflowX:"auto",maxHeight:320,overflowY:"auto"}}>
              <table style={{width:"100%",borderCollapse:"collapse",fontSize:12}}>
                <thead><tr style={{background:"#0c0e1a",position:"sticky",top:0}}>{["#","Row","A","B","C","D",""].map((h,i)=><th key={i} style={{padding:"6px 10px",color:"#252840",fontSize:9,letterSpacing:2,textTransform:"uppercase",borderBottom:"1px solid #1a1e35",textAlign:"center"}}>{h}</th>)}</tr></thead>
                <tbody>{rows.map((r,i)=><tr key={r.row} style={{background:i%2?"rgba(255,255,255,.01)":"transparent",borderBottom:"1px solid rgba(255,255,255,.02)"}}>
                  <td style={{padding:"5px 10px",color:"#252840",textAlign:"center"}}>{i+1}</td>
                  <td style={{padding:"5px 10px",color:"#fbbf24",fontWeight:700,textAlign:"center"}}>{pad2(r.row)}</td>
                  {COLS.map(col=><td key={col} style={{padding:"5px 10px",textAlign:"center",fontWeight:700,color:r[col]===null?"#1a1e35":CLR[col]}}>{r[col]===null?"—":pad2(r[col])}</td>)}
                  <td style={{padding:"5px 8px",textAlign:"center"}}><button onClick={()=>{setRows(prev=>prev.filter(x=>x.row!==r.row));st("Row "+r.row+" deleted","warn");}} style={{background:"transparent",border:"none",color:"#252840",cursor:"pointer",fontSize:11}}>✕</button></td>
                </tr>)}</tbody>
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
                      const tc=info.type==="custom"?"#f87171":info.type==="cross"?"#fbbf24":"#4a4e6a";
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
              {COLS.map(col=><NeuralScoreCol key={col} col={col} scores={S.weights[col]?S.weights[col].neuralScores||{}:{}}/>)}
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
                <div style={{display:"flex",gap:6}}>
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

      <div style={{position:"fixed",bottom:0,left:0,right:0,height:36,background:"#0c0e1a",borderTop:"1px solid #12152a",display:"flex",alignItems:"center",padding:"0 14px",gap:10,fontSize:10,zIndex:100}}>
        <div style={{width:7,height:7,borderRadius:"50%",flexShrink:0,background:stClr,boxShadow:"0 0 "+(msg.c==="busy"?8:5)+"px "+stClr}}/>
        <span style={{color:stClr}}>{msg.t}</span>
        <div style={{flex:1}}/>
        <span style={{color:"#1e2240"}}>{rows.length} rows · {accLog.length} sessions · {Object.keys(A).length+customs.length} algos · DB ✓</span>
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
  if(!entries.length)return <div style={{fontSize:9,color:"#2d3158"}}>No data yet</div>;
  return <div>
    <div style={{fontSize:9,color:CLR[p.col],letterSpacing:2,marginBottom:6}}>{p.col}</div>
    {entries.map(([name,score])=>{
      const clr=score>1?"#34d399":score>0?"#a78bfa":score<-0.5?"#f87171":"#4a4e6a";
      return <div key={name} style={{display:"flex",gap:5,alignItems:"center",marginBottom:3}}>
        <span style={{fontSize:9,color:"#4a4e6a",flex:1,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{name}</span>
        <span style={{fontSize:9,color:clr,fontWeight:700,minWidth:36,textAlign:"right"}}>{score>0?"+":""}{score.toFixed(2)}</span>
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
function Card(p){return <div style={{background:"#0c0e1a",border:"1px solid #1a1e35",borderRadius:10,padding:14,marginBottom:12,...(p.style||{})}}>{p.children}</div>;}
function SL(p){return <div style={{fontSize:9,letterSpacing:3,color:"#2d3158",textTransform:"uppercase",marginBottom:10,...(p.style||{})}}>{p.children}</div>;}
function FI(p){return <div><div style={{fontSize:9,color:p.color||"#4a4e6a",marginBottom:3,letterSpacing:2}}>{p.label}</div><input type="text" inputMode="numeric" maxLength={2} value={p.val} onChange={e=>p.onChange(e.target.value.replace(/\D/g,"").slice(0,2))} placeholder="00" style={{width:p.w||62,background:"#060709",border:"1px solid #1a1e35",color:"#c8d0e8",padding:"8px 6px",borderRadius:6,fontSize:14,fontFamily:"monospace",textAlign:"center",outline:"none"}}/></div>;}
function PB(p){return <button onClick={p.onClick} style={{background:"#a78bfa",border:"none",color:"#fff",padding:"9px 18px",borderRadius:6,cursor:"pointer",fontSize:12,fontFamily:"'Courier New',monospace",fontWeight:700,alignSelf:"flex-end",...(p.style||{})}}>{p.children}</button>;}
function GB(p){return <button onClick={p.onClick} style={{background:"transparent",border:"1px solid #1a1e35",color:"#8892b0",padding:"7px 12px",borderRadius:6,cursor:"pointer",fontSize:11,fontFamily:"'Courier New',monospace",alignSelf:"flex-end"}}>{p.children}</button>;}
