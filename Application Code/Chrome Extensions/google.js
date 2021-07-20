let x = document.getElementsByClassName("gLFyf gsfi");
let search = x[0].value;
var under_values = ["under", "below", "less", "min", "minimum"];
var over_values = ["above", "more", "greater", "max", "maximum"];
var between = ["between"];
let min = -1,
  max = -1;
let fla = search.search("laptop");
if (fla != -1) {
  var res = search.split(" ");
  console.log(res);
  var num = [];
  for (var i = 0; i < res.length; i++) {
    const reg = new RegExp("^[0-9]([0-9]{1,2})k$", "g");
    if (!isNaN(res[i])) {
      num.push(res[i]); //200k
      console.log(res[i]);
    } else if (reg.exec(res[i])) {
      let numm = parseInt(res[i].substring(0, res[i].length - 1)) * 1000;
      num.push(numm);
      console.log(numm);
    }
  }
  if (num.length == 1) {
    for (let j = 0; j < res.length; j++) {
      for (let i = 0; i < under_values.length; i++) {
        if (res[j] === under_values[i]) {
          max = num[0];
          break;
        }
      }
      if (max != -1) break;
    }
    if (max == -1) {
      for (let j = 0; j < res.length; j++) {
        for (let i = 0; i < over_values.length; i++) {
          if (res[j] === over_values[i]) {
            min = num[0];
          }
        }
        if (min != -1) break;
      }
    }
  } else if (num.length == 2) {
    if (parseInt(num[0]) > parseInt(num[1])) {
      max = num[0];
      min = num[1];
    } else {
      min = num[0];
      max = num[1];
    }
  }
  console.log(min + " " + max);
  var dict = [];
  dict.push({ minimum: min, maximum: max });
  var dictString = JSON.stringify(dict);
  chrome.storage.sync.set({ google_data: dictString }, function() {
    console.log("Success");
  });
}
