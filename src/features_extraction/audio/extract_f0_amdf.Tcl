#!/usr/bin/wish8.4

package require snack

snack::sound s

set f [open list/list.txt] 
set list [read $f] 
close $f

foreach file $list { 
 s read $file 

 set fd [open [file rootname $file].f0 w] 
 puts $fd [join [s pitch -method amdf -framelength 0.01 -minpitch 70 -maxpitch 300] \n] 
 close $fd 

 set fd [open [file rootname $file].frm w] 
 puts $fd [join [s formant] \n] 
 close $fd 
}

exit 
