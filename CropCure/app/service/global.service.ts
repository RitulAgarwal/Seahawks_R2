import { Injectable } from '@angular/core';
import { NavigationExtras, Router } from '@angular/router';

@Injectable({
  providedIn: 'root'
})
export class GlobalService {

  constructor(private router:Router) { }


  navigate(route: string) {
    this.router.navigate([route]);

  }

  navigateWithExtras(route: string, extras: NavigationExtras) {
    this.router.navigate([route], extras);
  }

  getNavigationExtras() {
    return this.router.getCurrentNavigation()?.extras?.state;
  }
}
