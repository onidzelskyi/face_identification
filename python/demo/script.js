/*
 Using flask endpoint
 */
function flask()
{
    var xmlhttp;
    if (window.XMLHttpRequest)
    {// code for IE7+, Firefox, Chrome, Opera, Safari
        xmlhttp=new XMLHttpRequest();
    }
    else
    {// code for IE6, IE5
        xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
    }
    xmlhttp.onreadystatechange=function()
    {
        if (xmlhttp.readyState==4 && xmlhttp.status==200)
        {
            // Text returned FROM the PHP script
            var response = xmlhttp.responseText;
            
            if(response) {
                // UPDATE ajaxTest content
                
                document.getElementById("myImage").setAttribute('src', 'data:image/jpg;base64,'+ response);
                document.getElementById("myImage").innerHTML = response;
                
                
            }         }
    }
    xmlhttp.open("GET","http://localhost:5000/",true);
    xmlhttp.send();
    document.getElementById("myImage").innerText="In porgress..";
}