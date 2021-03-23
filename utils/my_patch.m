function h=my_patch(vertex,face)


h=patch('vertices',vertex,'faces',face,'facecolor','g','edgecolor','none');

material shiny;
lighting phong;
axis equal;
