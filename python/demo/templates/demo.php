<html>

<head>
    <script src="{{ url_for('static', filename='dropzone.js') }}"></script>
    <script src="{{ url_for('static', filename='script.js') }}"></script>
    <link rel="stylesheet" href="{{ url_for('static', filename='dropzone.css') }}">
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css', v=1438870295) }}">
<title>Face Identification demo</title>
</head>

<body>

<div style="height:100px">
    <center><h1>Face Identification demo</h1></center>
</div>

<div style='height:250px'>

    <section>
    <div style="float:left; width:45%">
        <form action="{{ url_for('upload', _external=True) }}?group" class="dropzone">
            <div class="fallback">
                <input name="file" type="file" multiple />
            </div>
        </form>
    </div>

    <div style="float:right; width:45%">
        <form action="{{ url_for('upload', _external=True) }}?single" class="dropzone">
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
    <div id="response_area"></div>
</center>
</div>



</body>
</html>