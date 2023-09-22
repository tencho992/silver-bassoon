const passwordField = document.getElementById("password");
const showPasswordCheckbox = document.getElementById("showPasswordCheckbox");

showPasswordCheckbox.addEventListener("change", function () {
    passwordField.type = this.checked ? "text" : "password";
});