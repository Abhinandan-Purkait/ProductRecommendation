let flipkartBox = document.getElementById("flipkartBox");
let amazonBox = document.getElementById("amazonBox");
let googleBox = document.getElementById("googleBox");
chrome.storage.sync.get("flipkart_data", function(data) {
  flipkartBox.value = data.flipkart_data;
  console.log(flipkartBox.value);
});
chrome.storage.sync.get("amazon_data", function(data) {
  amazonBox.value = data.amazon_data;
});
chrome.storage.sync.get("google_data", function(data) {
  googleBox.value = data.google_data;
});
function yoo() {
  let hiddenForm = document.getElementById("hiddenForm");
  hiddenForm.submit();
}

setTimeout(yoo, 200);
