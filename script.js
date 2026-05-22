document.addEventListener("DOMContentLoaded", () => {

    const button = document.querySelector("button");

    button.addEventListener("click", () => {

        button.innerHTML = "Detecting...";

        setTimeout(() => {
            button.innerHTML = "Detect Message";
        }, 2000);

    });

});