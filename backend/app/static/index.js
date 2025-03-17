function imagePreview(event)
{
  var image=URL.createObjectURL(event.target.files[0]);
  var imagediv= document.getElementById('preview');
  var newimg=document.createElement('img');
  imagediv.innerHTML='';
  newimg.src=image;
  newimg.width="500";
  imagediv.appendChild(newimg);
}