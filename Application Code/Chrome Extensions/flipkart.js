let str = window.location.href;
let oldURL = "";
let MaxMinArray = [];
let currentURL = window.location.href;
let dict = [];
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function checkURLchange(currentURL) {

  if (currentURL != oldURL) {
    /*console.log("slept")
    await sleep(2000);
    console.log("awoke")*/
    str = currentURL;
    let pos=-1,pos1=-1
    pos = str.lastIndexOf("price_range.to%3D");
    let maxx = str.substring(pos + 17);
    if (maxx === "Max") maxx = "75000";
    pos1 = str.lastIndexOf("price_range.from%3D");
    let minn = str.substring(pos1 + 19, str.indexOf("&", pos1 + 19));
    if (minn === "Min") minn = "0";
    if(pos!=-1 && pos1!=-1)
    {
      MaxMinArray.push(maxx);
      MaxMinArray.push(minn);
      //let x = document.getElementsByClassName("_3wU53n");
      let y = document.getElementsByClassName("_1vC4OE _2rQ-NK");
      let z=document.getElementsByClassName('vFw0gD');
      for (let i = 0; i < y.length; i++)
      {
        //let sp = x[i].innerHTML;
        let pr = y[i].innerHTML.substring(1);
        let prInt = parseInt(pr, 10);
        let processor=z[i].childNodes[0].innerHTML;
        let ram=z[i].childNodes[1].innerHTML;
        let res=processor.split(" ");
        let flag=0;
        let ppp=res[0]+" "+res[1]+" "+res[2];
        for(let loop=0;loop<res.length;loop++)
        {
          let tp=res[loop]
          if(tp.toLowerCase()==="processor")
            flag++;

          if(tp.toLowerCase()=="ryzen" || tp.toLowerCase()=="amd")
            flag--;
          if(tp.charAt(0)=='(')
            ppp=ppp+" "+tp.charAt(1)
        }
        let reso=ram.split(" ");
        let ram_number=reso[0];
        for(let loop=0;loop<reso.length;loop++)
        {
          if(reso[loop]==="RAM")
            flag++;
        }
        minned = parseInt(minn, 10);
        maxxed = parseInt(maxx, 10);
        //console.log(minned+" "+maxxed);
        //if(prInt >= minned && prInt <=maxxed)
          //flag++;
        if(flag==2)
        {
          let b=pr.split(",")
          b=b.join("")
          dict.push({price: parseInt(b) ,processor: ppp ,ram: parseInt(ram_number)});
        }
        //console.log(processor+" "+ram)
        //dict.push({ specs: sp, price: pr });
      }

      oldURL = currentURL;
      console.log('URL CHANGED');
      console.log(dict);
    }
    let dictString = JSON.stringify(dict);
    chrome.storage.sync.set({ flipkart_data: dictString }, function() {
    console.log("Success");
    console.log(dictString);
    });
    //}
  }
  oldURL = window.location.href;
}
setInterval(function() {
  checkURLchange(window.location.href);
}, 10000);

checkURLchange(currentURL);

/*const reg = new RegExp("₹[0-9]");
let x = document.getElementsByClassName("_1vC4OE _3qQ9m1");
let y = document.getElementsByClassName("_35KyD6");
let matches = "";
for (let i = 0; i < x.length; i++) {
  if (x[i].innerHTML.match(reg)) {
    matches = matches + " " + x[i].innerHTML;
  }
}
console.log(matches);
if (y.length != 0) console.log(y[0].innerHTML);
*/
