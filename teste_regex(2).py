import re

with open('teste.txt','r') as file:
  cont = 1
  for linha in file:
    resultado = re.match(r'^(\d+[.]?\d+)(\s)(\d+[.]?\d+)',linha)
    
    if (cont == 1):
      coordA = [resultado.group(1),resultado.group(3)]
    else:
      coordB = [resultado.group(1),resultado.group(3)]
      
    if (cont == 2):
      coordF = [coordA,coordB]
      print(coordF)
      cont=1
    else:
      cont +=1
  

    
    
