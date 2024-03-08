
let windowWidth = outerWidth
let body = document.getElementById("body")
let measurement1 = document.getElementById("measurement1")




window.addEventListener('scroll', function() {
     // Get scroll progress (0 to 1)

    // Animation logic for measuring tapes based on scrollY
    measuringTapes.style.transform = `translateX(-${scrollY * 100}%)`; // Move tapes left based on scrollY (adjust multiplier for speed)
});

// Setup
body.style.transform = `translateX(-40%)`
measurement1.style.width = "45%"
measurement1.style.transform = "translateY(25px)"
measurement1.style.transform = "translateX(-90%)"
measurement1.style.transform = "scaleX(0.2)"


// Scroll
window.addEventListener('scroll', () => {
    const scrollY = window.scrollY / document.documentElement.scrollHeight;
    measurement1.style.transform = `translateX(${scrollY * 200 - 90}%)`;
    measurement1.style.transform = "scaleX(0.2)"

})