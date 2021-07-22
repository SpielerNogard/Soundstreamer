import mariadb

import config



class Database_Watcher(object):
    def __init__(self):
        self.get_config()

    def get_config(self):
        self.Server = config.server
        self.Username = config.username
        self.Password = config.password
        self.Databasename = config.databasename

        
        
    def print_das(self,nachricht):
        print(nachricht)

    def read_data(self,SQL):
        try:
            conn = mariadb.connect(
        user=self.Username,
        password=self.Password,
        host=self.Server,
        port=3306,
        database=self.Databasename

        )
            print("Information: Connected to Database: "+str(self.Databasename)+" with User: "+str(self.Username))
        except mariadb.Error as e:
            print(f"Error: Cant connecting to MariaDB Platform: {e}")
            sys.exit(1)
        cur = conn.cursor()
        print("Information: SQL command is executed: "+str(SQL))
        cur.execute(SQL) 
        Ergebnis1 = cur
        Ergebnis = []
        for a in Ergebnis1:
            Ergebnis.append(a)
        conn.close()
        print("Information: Connection to Database closed")
        return(Ergebnis)

    

    def write_data(self,SQL):
        try:
            conn = mariadb.connect(
        user=self.Username,
        password=self.Password,
        host=self.Server,
        port=3306,
        database=self.Databasename
        )
            print("Information: Connected to Database: "+str(self.Databasename)+" with User: "+str(self.Username))
        except mariadb.Error as e:
            print(f"Error: Cant connecting to MariaDB Platform: {e}")
            sys.exit(1)
        cur = conn.cursor()
        print("Information: SQL command is executed: "+str(SQL))
        cur.execute(SQL) 
        conn.commit() 
        conn.close()
        print("Information: Connection to Database closed")
        