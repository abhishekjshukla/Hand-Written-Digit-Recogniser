var clickX_simple = new Array();
var clickY_simple = new Array();
var clickDrag_simple = new Array();
var paint_simple;
var canvas_simple;
var context_simple;
var canvasWidth = 168;
var canvasHeight = 168;
var canvas;
var context;
var padding = 25;
var lineWidth = 8;
var colorPurple = "#cb3594";
var colorGreen = "#659b41";
var colorYellow = "#ffcf33";
var colorBrown = "#986928";
var outlineImage = new Image();
var crayonImage = new Image();
var markerImage = new Image();
var eraserImage = new Image();
var crayonBackgroundImage = new Image();
var markerBackgroundImage = new Image();
var eraserBackgroundImage = new Image();
var crayonTextureImage = new Image();
var clickX = new Array();
var clickY = new Array();
var clickColor = new Array();
var clickTool = new Array();
var clickSize = new Array();
var clickDrag = new Array();
var paint = false;
var curColor = colorPurple;
var curTool = "crayon";
var curSize = "normal";
var mediumStartX = 18;
var mediumStartY = 19;
var mediumImageWidth = 93;
var mediumImageHeight = 46;
var drawingAreaX = 111;
var drawingAreaY = 11;
var drawingAreaWidth = 267;
var drawingAreaHeight = 200;
var toolHotspotStartY = 23;
var toolHotspotHeight = 38;
var sizeHotspotStartY = 157;
var sizeHotspotHeight = 36;
var sizeHotspotWidthObject = new Object();
sizeHotspotWidthObject.huge = 39;
sizeHotspotWidthObject.large = 25;
sizeHotspotWidthObject.normal = 18;
sizeHotspotWidthObject.small = 16;
/**
* Creates a canvas element.
*/
function prepareSimpleCanvas()
{
	// Create the canvas (Neccessary for IE because it doesn't know what a canvas element is)
	var canvasDiv = document.getElementById('canvasSimpleDiv');
	canvas_simple = document.createElement('canvas');
	canvas_simple.setAttribute('width', canvasWidth);
	canvas_simple.setAttribute('height', canvasHeight);
	canvas_simple.setAttribute('id', 'canvasSimple');
	canvasDiv.appendChild(canvas_simple);
	if(typeof G_vmlCanvasManager != 'undefined') {
		canvas_simple = G_vmlCanvasManager.initElement(canvas_simple);
	}
	context_simple = canvas_simple.getContext("2d");
	context_simple.fillstyle="blue"
	context_simple.fillRect(0, 0, canvasWidth, canvasHeight);
	// Add mouse events
	// ----------------
	$('#canvasSimple').mousedown(function(e)
	{
		// Mouse down location
		var mouseX = e.pageX - this.offsetLeft;
		var mouseY = e.pageY - this.offsetTop;
		
		paint_simple = true;
		addClickSimple(mouseX, mouseY, false);
		redrawSimple();
	});
	
	$('#canvasSimple').mousemove(function(e){
		if(paint_simple){
			addClickSimple(e.pageX - this.offsetLeft, e.pageY - this.offsetTop, true);
			redrawSimple();
		}
	});
	
	$('#canvasSimple').mouseup(function(e){
		paint_simple = false;
	  	redrawSimple();
	});
	
	$('#canvasSimple').mouseleave(function(e){
		paint_simple = false;
	});
	
	$('#clearCanvasSimple').mousedown(function(e)
	{
		clickX_simple = new Array();
		clickY_simple = new Array();
		clickDrag_simple = new Array();
		clearCanvas_simple(); 
	});
}

function addClickSimple(x, y, dragging)
{
	clickX_simple.push(x);
	clickY_simple.push(y);
	clickDrag_simple.push(dragging);
}

function clearCanvas_simple()
{
	context_simple.clearRect(0, 0, canvasWidth, canvasHeight);
	context_simple = canvas_simple.getContext("2d");
	context_simple.fillstyle="#000000"
	context_simple.fillRect(0, 0, canvasWidth, canvasHeight);
}

function redrawSimple()
{
	clearCanvas_simple();
	
	var radius = 22;
	context_simple.strokeStyle = "#ffffff";
	context_simple.lineJoin = "round";
	context_simple.lineWidth = radius;
			
	for(var i=0; i < clickX_simple.length; i++)
	{		
		context_simple.beginPath();
		if(clickDrag_simple[i] && i){
			context_simple.moveTo(clickX_simple[i-1], clickY_simple[i-1]);
		}else{
			context_simple.moveTo(clickX_simple[i]-1, clickY_simple[i]);
		}
		context_simple.lineTo(clickX_simple[i], clickY_simple[i]);
		context_simple.closePath();
		context_simple.stroke();
	}
}
var dataURL;
var button = document.getElementById('btn-d');
button.addEventListener('click', function (e) {
   dataURL = canvas_simple.toDataURL();
   $.ajax({
    url: '/',
    data: {
    	data:dataURL
    	},
   	type: 'POST',
    success: function(response) {
    			window.location.reload();
            },
            error: function(error) {
                alert("fail");
            }

});
});

