<html>

<head>
    <script src="dropzone.js"></script>
    <script src="script.js"></script>
    <link rel="stylesheet" href="dropzone.css">
    <link rel="stylesheet" href="style.css?v=1438870295">
<title>Face Identification demo</title>
</head>

<body>

<div style="height:100px">
    <center><h1>Face Identification demo</h1></center>
</div>

<div style='height:250px'>

    <section>
    <div style="float:left; width:45%">
        <form action="http://localhost:5000/?group" class="dropzone">
            <div class="fallback">
                <input name="file" type="file" multiple />
            </div>
        </form>
    </div>

    <div style="float:right; width:45%">
        <form action="http://localhost:5000/?single" class="dropzone">
            <div class="fallback">
                <input name="file" type="file" multiple />
            </div>
        </form>
    </div>
</section>

</div>


<div style<='width:100%'>
<center>
    <button type="button" onclick="flask()">Request data</button>
</center>
</div>

<div id='response'>
</div>

<img id="myImage"/>

</body>
</html>