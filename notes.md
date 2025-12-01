вылетала вот такая очень стремная ошибка:
nvcc error : 'cudafe++' died with status 0xC0000005 (ACCESS_VIOLATION)

оказалось, что нужно использовать cl именно x64, а не x86. По этой причине в переменные среды добавил вот такой путь до cl.exe:
C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64

Теперь файлы CUDA компилируются!
nvcc first.cu

после этого создастся файл a.exe

запуск файла в терминале:
a


VSCODE:
C:\Users\Александр\AppData\Roaming\Code - тут лежат кэши для расширений vscode
C:\Users\Александр\.vscode\extensions - тут сами расширения