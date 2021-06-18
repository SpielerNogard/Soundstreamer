from datetime import datetime


class Auswerter(object):
    def __init__(self):
        f = open("Auswertung/Auswertung.txt", "w")
        f.write("")
        f.close()
        self.real_sende_times = []
        self.real_recieve_times = []
        self.unterschiede = []
        self.read_from_recievetimes()
        self.read_from_sendetimes()
        self.zeiten_erhalten()
        self.werte_aus()
    def read_from_sendetimes(self):
        with open("TimeStamps/ClientTimeStamps.txt") as f:
            content = f.readlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
        self.sendezeiten =[x.strip() for x in content] 

    def read_from_recievetimes(self):
        with open("TimeStamps/ServerTimeStamps.txt") as f:
            content = f.readlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
        self.erhaltezeiten =[x.strip() for x in content] 

    def zeiten_erhalten(self):
        for a in self.sendezeiten:
            print(a)
            test = datetime.strptime(a, "%Y-%m-%d %H:%M:%S.%f")
            self.real_sende_times.append(test)

        for a in self.erhaltezeiten:
            print(a)
            test = datetime.strptime(a, "%Y-%m-%d %H:%M:%S.%f")
            self.real_recieve_times.append(test)

    def werte_aus(self):
        lange_sende = len(self.real_sende_times)
        lange_recieve = len(self.real_recieve_times)
        print(lange_sende," ",lange_recieve)

        if lange_recieve == lange_sende:
            for i in range(0,lange_sende-1):
                Ausgang = self.real_sende_times[i]
                Eingang = self.real_recieve_times[i]
                duration = Eingang - Ausgang
                print(Eingang-Ausgang)
                self.unterschiede.append(duration.microseconds)
                Zeile = "gesendet: "+str(Ausgang)+" empfangen: "+str(Eingang)+" -----> vergangene Zeit: "+str(duration.seconds)+" Sekunden oder "+str(duration.microseconds)+" Microsekunden oder "+str(duration.microseconds/1000)+" ms"
                f = open("Auswertung/Auswertung.txt", "a")
                f.write(str(Zeile)+"\n")
                f.close()
                print(Ausgang-Eingang)

            lange_werte = len(self.unterschiede)
            ergebnis = 0
            hochste_dauer = 0
            for a in self.unterschiede:
                ergebnis = ergebnis+a
                if a > hochste_dauer:
                    hochste_dauer = a
            
            Zeile = "gesendetete Packete: "+str(lange_werte)+" insgesamt gebrauchte Zeit in microseconds: "+str(ergebnis)+" durchschnitt: "+str(ergebnis/lange_werte)+" microsecons oder "+str((ergebnis/lange_werte)/1000)+" ms höchste Dauer: "+str(hochste_dauer/1000)+" ms"
            f = open("Auswertung/Auswertung.txt", "a")
            f.write(str(Zeile)+"\n")
            f.close()



if __name__ == "__main__":
    CLAUS = Auswerter()
    