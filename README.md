# vitmav45-2017-DeepImpact

DeepImpact csapat
Csapattagok: Kiss Gábor

A repository tartalmazza a "Deep Learning a gyakorlatban python és Lua alapon" című tárgy keretein belül végzett házi feladatomat, amelynek célja olyan parancsszófelismerésre alkalmas neurális hálózat tervezése és tanítása volt, amely alkalmas lehet robotok hangvezérlő moduljának alapjául szolgálni. 

Futtatáshoz szükséges modulok:
A scripteket python 3.6-ban készítettem Keras keretrendszerrel, Tensorflow backend használatával.
A megoldások során használt modulok: numpy, sklearn, scipy, python_speech_features, glob2
Telepítéshez futtatandó:
sudo apt-get install python-numpy python-scipy python-sklearn glob2
pip install python_speech_features

Az adatlőkészítés során használt forrás fájlok és a tanításhoz és teszteléshez használt fájlok az alábbi linken elérhetőek: https://www.dropbox.com/sh/4tj64gl41rkxpje/AACKu0lumg_Q74UwlMijFZ-Ta?dl=0

Programok használata:
Tanító scriptek:
A tanításhoz használt scriptek mindegyik modellhez a /TrainerScripts/ mappában találhatóak. Futtatásuk 4 paraméter megadásával történik. Első paraméter a tanító adatokat tartalmazó fájl elérési útja, a második a teszt adatokat tartalmazó fájl elérési útja, a harmadik a validációs adatokat tartalmazó fájl elérési útja és a negyedik paraméter a tanított modell kívánt elérési útja fájlnévvel együtt, lehetőleg hdf5 kiterjesztéssel. Például:
gabor@gabor-Lenovo-IdeaPad-Z500:~$ python /home/gabor/DeepLearningHomework/vitmav45-2017-DeepImpact/TrainerScripts/CNN+Dense_sep.py /home/gabor/DeepLearningHomework/vitmav45-2017-DeepImpact/traindata_sep.npy /home/gabor/DeepLearningHomework/vitmav45-2017-DeepImpact/testdata_sep.npy /home/gabor/DeepLearningHomework/vitmav45-2017-DeepImpact/validdata_sep.npy /home/gabor/DeepLearningHomework/valami.hdf5

A tanítás során a program először beolvassa a fájlokból az értékeket, majd létrehozza a modellt, amit leprogramoztam. Ezt követően elkezdi a tanítást, amit figyelemmel kísérhetünk a terminál ablakban. 1000 epoch után vagy early stopping-al leáll a tanítás és a rendszer a legjobb validációs költségű modellt visszatölti és kiértékeli. A kiértékelés során a teszt adatokra kiszámolja a költséget, valamint minden szóra a precizitás (precision), felidézés (recall) és f1 paramétereket. Ez után a konfúziós mátrix alapján kiszámítja a pontosságot (accuracy), mint a főátlóban lévő elemek és az összes elem hányadosa. Ezt követően a modell teljes leírását is kiírja a képernyőre. 

Kiértékelő script:
Egy hálózat kiértékelése a tester.py script segítségével történhet. Ez a srcipt is egyszerűen futtatható prancssorból python segítségével. Kettő paramétert kell megadnunk a működéshez. Az első a tesztelni kívánt modellnek az elérési útja, a második pedig a megfelelő teszt adathalmazt tartalmazó fájl elérési útja. Például: 
gabor@gabor-Lenovo-IdeaPad-Z500:~$ python /home/gabor/DeepLearningHomework/vitmav45-2017-DeepImpact/tester.py /home/gabor/DeepLearningHomework/vitmav45-2017-DeepImpact/Models/weights_CNN+RNN+Dense_sep_MFCC.hdf5 /home/gabor/Audio/testdata_endtoend_sep_nofilter.npy

Adatelőkészítő sriptek:
Az adatelőkészítő scriptek a /DataPreprocessers/ mappában találhatóak. 3 különböző típus van belőlük. A "data_preprocesser.py" beolvas minden hangfájlt, MFCC transzformációt hajt rajtuk végre a dokumentációban megfogalmazottak alapján, majd összekeveri őket és tanító, validációs és teszt adatokra bontja szét őket. Ezt követően a tanító adatok alapján nulla várható értékű, egységnyi szórásúvá alakítja az adathalmazokat. Ez után mind a három adathalmazt labelekkel együtt kimenti fájlokba.
A "data_preprocesser_sep.py" script hasonlóan működik, mint a fent leírt program azzal a különbséggel, hogy külön beszélőktől felvett hangfájlokból alakítja ki a tanító, teszt és validációs adathalmazokat, tehát egy beszélő összes felvétele ugyan abba a csoportba kerül. A szeparált és normalizálrt adatokat ez is három külön fájlba menti ki.
A "data_preprocesser_endtoend_sep.py" ugyan úgy működik, mint a "data_preprocesser_sep.py", azonban MFCC transzformáció helyett fix ablakszélességű szeletekre bontja a felvételeket fix lépésközzel. Ezt követően ugyan úgy normálja őket a tanító adatok alapján és kimenti őket 3 külön fájlba.
Az adatelőkészítő scripteknek 6 paramétere van, rendre a tanításra szánt hangfájlokat tartalmazó mappa elérési útja, a tesztelésre szánt hangfájlokat tartalmazó mappa elérési útja, a validációra szánt hangfájlokat tartalmazó mappa elérési útja, az előkészített tanító adatsor kívánt elérési útja fájlnévvel együtt, az előkészített teszt adatsor kívánt elérési útja fájlnévvel együtt, az előkészített validációs adatsor kívánt elérési útja fájlnévvel együtt, utóbbi három esetben lehetőleg .npy kiterjesztéssel. 

A repository tartalmazza továbbá a dokumentációban is elemzett, betanított modelleket a /Models/ mappában, a hozzájuk tartozó Keras summary-ket és teszt eredményeket a "ModelSummaries&TestResults.txt" fájlban, az egyes modellekhez tartozó tévesztési mátrixokat a /ConfusionMatrices/ mappában, a dokumentációt szerkeszteheto doc és pdf formátumban, valamint a LabVIEW programot, aminek segítségével a felvételek készültek.

